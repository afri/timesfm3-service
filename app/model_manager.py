import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.config import settings

logger = logging.getLogger("timesfm3.model")


class TimesFMModelManager:
    """Manages loading and inference of TimesFM forecasting models."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TimesFMModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = None
        self.model_id = settings.MODEL_ID
        self.device = self._resolve_device(settings.DEVICE)
        self.backend = "unknown"
        self._infer_lock = threading.Lock()
        self._initialized = True
        logger.info(f"TimesFMModelManager initialized (device={self.device})")

    def _resolve_device(self, preferred_device: str) -> str:
        if preferred_device.lower() != "auto":
            return preferred_device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self) -> None:
        """Loads the TimesFM model from checkpoint or Hugging Face Hub."""
        with self._infer_lock:
            if self.model is not None:
                return

            logger.info(f"Loading TimesFM model '{self.model_id}' on device '{self.device}'...")
            t0 = time.time()

            try:
                # 1. Try TimesFM 3.0 API
                try:
                    import timesfm3
                    if hasattr(timesfm3, "TimesFM3Forecaster"):
                        logger.info("Initializing TimesFM3Forecaster...")
                        self.model = timesfm3.TimesFM3Forecaster.from_pretrained(
                            self.model_id,
                            device=self.device,
                        )
                        self.backend = "timesfm3"
                        logger.info(f"Successfully loaded TimesFM 3 model in {time.time() - t0:.2f}s")
                        return
                except (ImportError, AttributeError) as e:
                    logger.debug(f"TimesFM3 API not available: {e}")

                # 2. Try standard timesfm package (TimesFM 2.5 / 3.0)
                import timesfm
                import torch

                if hasattr(timesfm, "TimesFM3"):
                    self.model = timesfm.TimesFM3.from_pretrained(self.model_id)
                    self.backend = "timesfm-3"
                elif hasattr(timesfm, "TimesFM_2p5_200M_torch"):
                    logger.info(f"Loading TimesFM_2p5_200M_torch from {self.model_id}...")
                    self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
                    if hasattr(self.model, "compile"):
                        self.model.compile(
                            timesfm.ForecastConfig(
                                max_context=settings.MAX_CONTEXT,
                                max_horizon=settings.MAX_HORIZON,
                                normalize_inputs=settings.NORMALIZE_INPUTS,
                            )
                        )
                    self.backend = "timesfm-2.5-torch"
                elif hasattr(timesfm, "TimesFmCheckpoint"):
                    tfm_ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.model_id)
                    self.model = timesfm.TimesFm(
                        hparams=timesfm.TimesFmHparams(
                            context_len=settings.MAX_CONTEXT,
                            horizon_len=settings.MAX_HORIZON,
                        ),
                        checkpoint=tfm_ckpt,
                    )
                    self.backend = "timesfm-v1"
                else:
                    raise RuntimeError(f"Compatible TimesFM class not found in timesfm module: {dir(timesfm)}")

                logger.info(f"Successfully loaded model '{self.model_id}' via {self.backend} in {time.time() - t0:.2f}s")

            except Exception as e:
                logger.error(f"Failed to load model '{self.model_id}': {e}", exc_info=True)
                raise RuntimeError(f"Error loading TimesFM model: {e}")

    def forecast(
        self,
        series: Union[List[float], List[List[float]], List[List[List[float]]]],
        horizon: Optional[int] = None,
        return_quantiles: bool = False,
        use_symmetric_averaging: bool = False,
        past_only_covariates: Optional[Any] = None,
        past_future_covariates: Optional[Any] = None,
    ) -> Tuple[Any, Optional[Any], float]:
        """Runs forecasting on input series. Returns (point_forecast, quantiles_data, elapsed_ms)."""
        if not self.is_loaded:
            self.load_model()

        h = horizon or settings.DEFAULT_HORIZON
        t0 = time.time()

        with self._infer_lock:
            try:
                point_res, quantiles_res = self._run_model_inference(
                    series=series,
                    horizon=h,
                    return_quantiles=return_quantiles,
                    use_symmetric_averaging=use_symmetric_averaging,
                    past_only_covariates=past_only_covariates,
                    past_future_covariates=past_future_covariates,
                )
                elapsed_ms = (time.time() - t0) * 1000.0
                return point_res, quantiles_res, elapsed_ms
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid forecasting input: {e}")
                raise
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                raise RuntimeError(f"Model forecasting failed: {e}")

    def _prepare_covariates(
        self,
        covariates: Optional[Any],
        is_single: bool,
    ) -> Optional[List[np.ndarray]]:
        if covariates is None:
            return None

        # Check for empty structures: [], [[]], etc.
        if isinstance(covariates, (list, tuple, np.ndarray)):
            if len(covariates) == 0:
                return None
            if len(covariates) == 1 and isinstance(covariates[0], (list, tuple, np.ndarray)) and len(covariates[0]) == 0:
                return None

        if is_single:
            arr = np.asarray(covariates, dtype=np.float32)
            if arr.size == 0:
                return None
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            elif arr.ndim != 2:
                raise ValueError(
                    f"Covariates for a single series must be 1D or 2D (num_channels, time), got {arr.ndim}D shape {arr.shape}."
                )
            return [arr]
        else:
            if not isinstance(covariates, (list, tuple, np.ndarray)):
                raise ValueError("Covariates for batch forecasting must be a list with one entry per series.")
            cov_list = []
            for c in covariates:
                arr = np.asarray(c, dtype=np.float32)
                if arr.size == 0:
                    raise ValueError("Covariates entries cannot be empty.")
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                elif arr.ndim != 2:
                    raise ValueError(
                        f"Covariates per series must be 1D or 2D (num_channels, time), got {arr.ndim}D shape {arr.shape}."
                    )
                cov_list.append(arr)
            return cov_list

    def _run_model_inference(
        self,
        series: Any,
        horizon: int,
        return_quantiles: bool = False,
        use_symmetric_averaging: bool = False,
        past_only_covariates: Optional[Any] = None,
        past_future_covariates: Optional[Any] = None,
    ) -> Tuple[Any, Optional[Any]]:
        # Normalize input to numpy batch
        is_single = False
        if isinstance(series, list) and len(series) > 0 and isinstance(series[0], (int, float)):
            input_list = [np.array(series, dtype=np.float32)]
            is_single = True
        elif isinstance(series, list) and len(series) > 0 and isinstance(series[0], list):
            input_list = [np.array(s, dtype=np.float32) for s in series]
        else:
            input_list = [np.asarray(series, dtype=np.float32)]

        # Prepare and validate covariates
        past_only_list = self._prepare_covariates(past_only_covariates, is_single)
        past_future_list = self._prepare_covariates(past_future_covariates, is_single)

        if past_only_list is not None:
            if len(past_only_list) != len(input_list):
                raise ValueError(
                    f"Number of past_only_covariates ({len(past_only_list)}) does not match number of series ({len(input_list)})."
                )
            for idx, (s_arr, c_arr) in enumerate(zip(input_list, past_only_list)):
                s_len = s_arr.shape[-1]
                c_len = c_arr.shape[-1]
                if c_len != s_len:
                    raise ValueError(
                        f"past_only_covariates length ({c_len}) for series index {idx} does not match context length ({s_len})."
                    )

        if past_future_list is not None:
            if len(past_future_list) != len(input_list):
                raise ValueError(
                    f"Number of past_future_covariates ({len(past_future_list)}) does not match number of series ({len(input_list)})."
                )
            for idx, (s_arr, c_arr) in enumerate(zip(input_list, past_future_list)):
                s_len = s_arr.shape[-1]
                c_len = c_arr.shape[-1]
                expected_len = s_len + horizon
                if c_len != expected_len:
                    raise ValueError(
                        f"past_future_covariates length ({c_len}) for series index {idx} does not match expected length context_len + horizon ({s_len} + {horizon} = {expected_len}). TimesFM 3 requires past_future_covariates to cover both past context and future horizon."
                    )

        # Run forecast based on backend
        if self.backend == "timesfm3" or hasattr(self.model, "predict_batch"):
            predict_kwargs: Dict[str, Any] = {
                "contexts": input_list,
                "horizon": horizon,
                "return_quantiles": return_quantiles,
                "use_symmetric_averaging": use_symmetric_averaging,
            }
            if past_only_list is not None:
                predict_kwargs["past_only_covariates"] = past_only_list
            if past_future_list is not None:
                predict_kwargs["past_future_covariates"] = past_future_list

            results = list(self.model.predict_batch(**predict_kwargs))

            point_batch = []
            quantiles_batch = []

            for out in results:
                f_arr = out.forecast
                point_batch.append(f_arr.tolist() if hasattr(f_arr, "tolist") else f_arr)

                if return_quantiles and out.quantiles is not None:
                    q_arr = out.quantiles
                    quantiles_batch.append(q_arr.tolist() if hasattr(q_arr, "tolist") else q_arr)

            final_point = point_batch[0] if is_single else point_batch
            final_q = None
            if return_quantiles and len(quantiles_batch) > 0:
                final_q = quantiles_batch[0] if is_single else quantiles_batch

            return final_point, final_q

        elif hasattr(self.model, "forecast"):
            # TimesFM 2.5 or standard API
            try:
                point_arr, q_arr = self.model.forecast(inputs=input_list, horizon=horizon)
            except TypeError:
                point_arr, q_arr = self.model.forecast(input_list, horizon=horizon)

            if hasattr(point_arr, "tolist"):
                point_list = point_arr.tolist()
            else:
                point_list = point_arr

            if is_single and isinstance(point_list, list) and len(point_list) == 1:
                point_list = point_list[0]

            final_q = None
            if return_quantiles and q_arr is not None:
                final_q = q_arr.tolist() if hasattr(q_arr, "tolist") else q_arr
                if is_single and isinstance(final_q, list) and len(final_q) == 1:
                    final_q = final_q[0]

            return point_list, final_q
        else:
            raise RuntimeError("Model does not expose a callable predict_batch or forecast method")


# Global model manager accessor
model_manager = TimesFMModelManager()

