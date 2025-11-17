# agents/orchestrator.py
import asyncio
import time
import traceback
from typing import Any, Awaitable, Callable, Optional, Tuple

from agents.schemas import Processed, OrchestratorOutput
from agents.behavior_agent import behavior_agent
from agents.safety_agent import safety_agent_with_gps
from agents.advise_agent import advise_agent
from nlg.llm_runtime_http import LLMRuntimeHTTP
from utils.metrics import RowMetrics


# Tipo de callback para quem quiser receber o resultado do LLM
# on_llm_result(row_id, message, source, meta, snapshot)
OnLLMResultCallback = Callable[
    [int, str, str, dict[str, Any], dict[str, Any]],
    Awaitable[None],
]


class Orchestrator:
    """
    Orquestra os agentes de comportamento, segurança e (indiretamente) linguagem.

    Responsabilidades:
      - No tick (run_once):
          * chama behavior_agent
          * chama safety_agent_with_gps
          * mede tempos via RowMetrics
          * NÃO chama o LLM (advise_agent com self.llm) aqui
      - Em background:
          * mantém uma fila interna de jobs de LLM
          * aplica rate-limit global (llm_min_interval_s)
          * chama advise_agent(policy, alerts, self.llm) no worker
          * opcionalmente dispara um callback com o resultado (para CSV / WebSocket)
    """

    def __init__(
        self,
        llm: Optional[LLMRuntimeHTTP],
        *,
        llm_min_interval_s: float = 12.0,
        on_llm_result: Optional[OnLLMResultCallback] = None,
    ) -> None:
        """
        Args:
            llm: runtime de LLM, usado apenas no worker interno.
            llm_min_interval_s: intervalo mínimo entre chamadas reais ao LLM.
            on_llm_result: callback assíncrono chamado quando o LLM devolver algo.
                           Ideal para atualizar CSV + broadcast fora deste módulo.
        """
        self.llm: Optional[LLMRuntimeHTTP] = llm
        self.llm_min_interval_s: float = float(llm_min_interval_s)
        self.on_llm_result: Optional[OnLLMResultCallback] = on_llm_result

        # Fila interna de jobs de LLM:
        # cada item é (row_id, policy, alerts, snapshot_dict)
        self._llm_queue: "asyncio.Queue[Tuple[int, Any, Any, dict[str, Any]]]" = (
            asyncio.Queue()
        )

        # Estado para rate-limit global
        self._last_llm_ts: Optional[float] = None

        # Task de worker em background
        self._worker_task: Optional[asyncio.Task] = None

    # -------------------------------------------------------------------------
    # Parte 1: Tick principal (sem LLM)
    # -------------------------------------------------------------------------

    async def run_once(self, processed: Processed) -> OrchestratorOutput:
        """
        Executa um 'tick' de orquestração:
          - chama behavior_agent
          - chama safety_agent_with_gps
          - mede tempos via RowMetrics
          - NÃO chama o LLM aqui (apenas agentes determinísticos)
        """
        try:
            rec = RowMetrics()

            with rec.block("agent.behavior"):
                policy = await behavior_agent(processed)

            with rec.block("agent.safety_gps"):
                alerts = await safety_agent_with_gps(
                    speed_kmh=processed.speed,
                    lat=processed.latitude,
                    lon=processed.longitude,
                    radius_m=500,  # ajuste se necessário
                )

            # IMPORTANTE:
            # Não chamamos advise_agent(policy, alerts, self.llm) aqui,
            # para não gerar 1 chamada de LLM por segundo.
            # Se quiser um fallback determinístico, você pode:
            #   message_fallback = await advise_agent(policy, alerts, None)
            # Por enquanto deixamos None para evitar surpresas.
            message = None

            return OrchestratorOutput(
                policy=policy,
                alerts=alerts,
                message=message,
                metrics=rec.as_flat(),  # m.agent.behavior.*, m.agent.safety_gps.*
            )

        except Exception as e:
            print("\n[Orchestrator] ERRO em run_once:", type(e).__name__)
            print(traceback.format_exc())
            raise

    # -------------------------------------------------------------------------
    # Parte 2: API pública para LLM em background
    # -------------------------------------------------------------------------

    async def start_background_tasks(self) -> None:
        """
        Inicia o worker interno de LLM.
        Deve ser chamado uma vez na inicialização da aplicação.
        """
        if self.llm is None:
            # Sem LLM configurado, não faz sentido iniciar worker.
            return

        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._llm_worker_loop())

    async def stop_background_tasks(self) -> None:
        """
        Cancela o worker interno de LLM, se estiver rodando.
        Útil em shutdown limpo da aplicação.
        """
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def enqueue_llm_job(
        self,
        row_id: int,
        policy: Any,
        alerts: Any,
        snapshot: dict[str, Any],
        *,
        force: bool = False,
    ) -> None:
        """
        Enfileira um job de LLM (row_id, policy, alerts, snapshot), respeitando o rate-limit.

        Args:
            row_id: chave única da linha no CSV (ou outro identificador externo).
            policy: objeto de policy retornado pelo behavior_agent.
            alerts: lista/estrutura de alerts retornada pelo safety_agent.
            snapshot: dicionário com o estado bruto (Processed serializado).
            force: se True, ignora o rate-limit de tempo (use com cuidado).
        """
        if self.llm is None:
            return

        now = time.monotonic()

        # Aplica rate-limit global, exceto se force=True
        if not force and self._last_llm_ts is not None:
            elapsed = now - self._last_llm_ts
            if elapsed < self.llm_min_interval_s:
                # Intervalo mínimo não atingido, não enfileira
                return

        await self._llm_queue.put((row_id, policy, alerts, dict(snapshot)))
        self._last_llm_ts = now

    # -------------------------------------------------------------------------
    # Parte 3: Worker interno de LLM
    # -------------------------------------------------------------------------

    async def _llm_worker_loop(self) -> None:
        """
        Loop interno que consome a fila de LLM, chama advise_agent com self.llm,
        e (opcionalmente) entrega o resultado para o callback on_llm_result.

        NÃO conhece CSV, WebSocket ou outros detalhes de I/O.
        Isso fica a cargo do callback, se configurado.
        """
        import random

        assert self.llm is not None, "llm_worker_loop iniciado sem self.llm configurado"

        while True:
            row_id, policy, alerts, snap = await self._llm_queue.get()
            try:
                final_msg: str = ""
                final_src: str = "error"
                final_meta: dict[str, Any] = {}

                # 1) Tenta chamar o LLM algumas vezes
                attempts = 0
                while attempts < 3:
                    attempts += 1
                    try:
                        ret = await advise_agent(policy, alerts, self.llm)

                        # advise_agent pode retornar:
                        #   (msg, src) ou (msg, src, meta)
                        if isinstance(ret, tuple):
                            if len(ret) == 2:
                                msg, src = ret
                                meta = {}
                            elif len(ret) >= 3:
                                msg, src, meta = ret[0], ret[1], (ret[2] or {})
                            else:
                                msg, src, meta = "", "error", {}
                        else:
                            msg, src, meta = str(ret), "model", {}

                    except Exception:
                        # Qualquer exceção aqui conta como tentativa falha
                        msg, src, meta = "", "error", {}
                        # Pequeno backoff antes de tentar de novo
                        await asyncio.sleep(0.5 * attempts + random.uniform(0.0, 0.25))

                    # Decide se aceita o resultado
                    if src == "model" and (msg or "").strip():
                        final_msg = str(msg)
                        final_src = str(src)
                        final_meta = meta or {}
                        break
                    else:
                        # Guarda o último erro/ fallback para não perder informação
                        final_msg = str(msg or "")
                        final_src = str(src or "error")
                        final_meta = meta or {}
                        # Se ainda há tentativas, espera um pouco
                        if attempts < 3:
                            await asyncio.sleep(
                                0.5 * attempts + random.uniform(0.0, 0.25)
                            )

                # 2) Callback para o consumidor (CSV / WebSocket / logs, etc.)
                if self.on_llm_result is not None:
                    try:
                        await self.on_llm_result(
                            row_id,
                            final_msg,
                            final_src,
                            final_meta,
                            snap,
                        )
                    except Exception:
                        print(
                            "[Orchestrator] erro no on_llm_result callback "
                            f"para row_id={row_id}"
                        )
                        print(traceback.format_exc())

                # Log simples opcional
                print(
                    f"[Orchestrator.llm_worker] row_id={row_id} src={final_src} "
                    f"msg_len={len(final_msg or '')}"
                )

            except Exception:
                print("[Orchestrator.llm_worker] erro inesperado:")
                print(traceback.format_exc())

            finally:
                self._llm_queue.task_done()