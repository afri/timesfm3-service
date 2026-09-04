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
        self.mock_mode = settings.MOCK_MODE
        self.backend = "unknown"
        self._infer_lock = threading.Lock()
        self._initialized = True
        logger.info(f"TimesFMModelManager initialized (device={self.device}, mock={self.mock_mode})")

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
        return (self.model is not None) or self.mock_mode

    def load_model(self) -> None:
        """Loads the TimesFM model from checkpoint or Hugging Face Hub."""
        if self.mock_mode:
            logger.info("Running in MOCK_MODE: Skipping heavy model weights download.")
            self.backend = "mock"
            return

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
        quantiles: Optional[List[float]] = None,
        past_covariates: Optional[Dict[str, Any]] = None,
        future_covariates: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Optional[Dict[str, Any]], float]:
        """Runs forecasting on input series. Returns (point_forecast, quantiles_dict, elapsed_ms)."""
        if not self.is_loaded:
            self.load_model()

        h = horizon or settings.DEFAULT_HORIZON
        t0 = time.time()

        # Handle Mock Mode
        if self.mock_mode or self.backend == "mock":
            point, q_dict = self._mock_forecast(series, h, quantiles)
            elapsed_ms = (time.time() - t0) * 1000.0
            return point, q_dict, elapsed_ms

        with self._infer_lock:
            try:
                point_res, quantiles_res = self._run_model_inference(
                    series=series,
                    horizon=h,
                    quantiles=quantiles,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                )
                elapsed_ms = (time.time() - t0) * 1000.0
                return point_res, quantiles_res, elapsed_ms
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                raise RuntimeError(f"Model forecasting failed: {e}")

    def _run_model_inference(
        self,
        series: Any,
        horizon: int,
        quantiles: Optional[List[float]],
        past_covariates: Optional[Dict[str, Any]],
        future_covariates: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        # Normalize input to numpy batch
        is_single = False
        if isinstance(series, list) and len(series) > 0 and isinstance(series[0], (int, float)):
            input_list = [np.array(series, dtype=np.float32)]
            is_single = True
        elif isinstance(series, list) and len(series) > 0 and isinstance(series[0], list):
            input_list = [np.array(s, dtype=np.float32) for s in series]
        else:
            input_list = [np.asarray(series, dtype=np.float32)]

        # Run forecast based on backend
        if self.backend == "timesfm3" or hasattr(self.model, "predict_batch"):
            return_q = quantiles is not None
            results = list(
                self.model.predict_batch(
                    contexts=input_list,
                    horizon=horizon,
                    return_quantiles=return_q,
                )
            )

            model_q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            target_q = quantiles or model_q_levels

            point_batch = []
            quantiles_batch = []

            for out in results:
                f_arr = out.forecast
                point_batch.append(f_arr.tolist() if hasattr(f_arr, "tolist") else f_arr)

                if return_q and out.quantiles is not None:
                    q_arr = out.quantiles
                    series_q = {}
                    for q in target_q:
                        closest_idx = int(np.argmin([abs(q - mq) for mq in model_q_levels]))
                        q_key = f"q{int(round(q * 100))}"
                        if q_arr.ndim == 2:
                            series_q[q_key] = q_arr[:, closest_idx].tolist()
                        elif q_arr.ndim == 3:
                            series_q[q_key] = q_arr[..., closest_idx].tolist()
                        else:
                            series_q[q_key] = q_arr.tolist()
                    quantiles_batch.append(series_q)

            final_point = point_batch[0] if is_single else point_batch
            final_q = None
            if return_q and len(quantiles_batch) > 0:
                final_q = quantiles_batch[0] if is_single else {"batch": quantiles_batch}

            return final_point, final_q

        elif hasattr(self.model, "forecast"):
            # TimesFM 2.5 or standard API
            try:
                point_arr, q_arr = self.model.forecast(inputs=input_list, horizon=horizon)
            except TypeError:
                point_arr, q_arr = self.model.forecast(input_list, horizon=horizon)
        else:
            raise RuntimeError("Model does not expose a callable predict_batch or forecast method")

        # Convert outputs to Python native structures
        if hasattr(point_arr, "tolist"):
            point_list = point_arr.tolist()
        else:
            point_list = point_arr

        if is_single and isinstance(point_list, list) and len(point_list) == 1:
            point_list = point_list[0]

        q_dict = None
        if q_arr is not None:
            if hasattr(q_arr, "tolist"):
                q_data = q_arr.tolist()
            else:
                q_data = q_arr
            if is_single and isinstance(q_data, list) and len(q_data) == 1:
                q_data = q_data[0]
            q_dict = {"quantiles": q_data}

        return point_list, q_dict

    def _mock_forecast(
        self,
        series: Any,
        horizon: int,
        quantiles: Optional[List[float]],
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Synthetic forecasting generator for testing and demonstration."""
        is_single = False
        if isinstance(series, list) and len(series) > 0 and isinstance(series[0], (int, float)):
            batch = [series]
            is_single = True
        elif isinstance(series, list) and len(series) > 0 and isinstance(series[0], list):
            batch = series
        else:
            batch = [[float(x) for x in series]]

        point_batch = []
        quantiles_batch = []

        target_quantiles = quantiles or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for s in batch:
            last_val = float(s[-1]) if len(s) > 0 else 0.0
            prev_val = float(s[-2]) if len(s) > 1 else last_val
            trend = (last_val - prev_val) * 0.5

            preds = []
            cur = last_val
            for step in range(horizon):
                cur += trend + 0.1 * np.sin(step)
                preds.append(round(cur, 4))
            point_batch.append(preds)

            # Generate synthetic quantiles
            series_q = {}
            for q in target_quantiles:
                factor = (q - 0.5) * 2.0  # -1.0 to 1.0
                q_vals = [round(p + factor * (0.05 * abs(p) + 0.5), 4) for p in preds]
                series_q[f"q{int(q*100)}"] = q_vals
            quantiles_batch.append(series_q)

        final_point = point_batch[0] if is_single else point_batch
        final_q = quantiles_batch[0] if is_single else {"batch": quantiles_batch}

        return final_point, final_q


# Global model manager accessor
model_manager = TimesFMModelManager()
