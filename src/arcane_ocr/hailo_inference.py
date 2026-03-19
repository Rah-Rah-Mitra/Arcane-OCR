from __future__ import annotations

import os
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm
from hailo_platform.pyhailort.pyhailort import FormatOrder


class HailoInfer:
    """Minimal async inference wrapper for a single HEF model."""

    def __init__(
        self,
        hef_path: str,
        batch_size: int = 1,
        input_type: Optional[str] = None,
        output_type: Optional[str] = None,
        priority: int = 0,
        vdevice: Optional[VDevice] = None,
        scheduler_algorithm: HailoSchedulingAlgorithm = HailoSchedulingAlgorithm.ROUND_ROBIN,
        group_id: str = "SHARED",
    ) -> None:
        if vdevice is None:
            params = VDevice.create_params()
            params.scheduling_algorithm = scheduler_algorithm
            params.group_id = group_id
            self.target = VDevice(params)
        else:
            self.target = vdevice

        hef_path = os.fspath(hef_path)
        self.hef = HEF(hef_path)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)

        self._set_input_type(input_type)
        self._set_output_type(output_type)

        self.config_ctx = self.infer_model.configure()
        self.configured_model = self.config_ctx.__enter__()
        self.configured_model.set_scheduler_priority(priority)
        self.last_infer_job = None

    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        if input_type is not None:
            self.infer_model.input().set_format_type(getattr(FormatType, input_type))

    def _set_output_type(self, output_type: Optional[str] = None) -> None:
        self.nms_postprocess_enabled = False

        if self.infer_model.outputs[0].format.order == FormatOrder.HAILO_NMS_WITH_BYTE_MASK:
            self.nms_postprocess_enabled = True
            self.output_type = self._output_data_type2dict("UINT8")
            return

        self.output_type = self._output_data_type2dict(output_type)
        for name, dtype in self.output_type.items():
            self.infer_model.output(name).set_format_type(getattr(FormatType, dtype))

    def get_vstream_info(self) -> Tuple[list, list]:
        return self.hef.get_input_vstream_infos(), self.hef.get_output_vstream_infos()

    def get_input_shape(self) -> Tuple[int, ...]:
        return self.hef.get_input_vstream_infos()[0].shape

    def run(self, input_batch: List[np.ndarray], inference_callback_fn: Callable):
        bindings_list = self._create_bindings(self.configured_model, input_batch)
        self.configured_model.wait_for_async_ready(timeout_ms=10000)
        self.last_infer_job = self.configured_model.run_async(
            bindings_list,
            partial(inference_callback_fn, bindings_list=bindings_list),
        )
        return self.last_infer_job

    def _create_bindings(self, configured_model, input_batch: List[np.ndarray]):
        def _frame_binding(frame: np.ndarray):
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=getattr(np, self.output_type[name].lower()),
                )
                for name in self.output_type
            }
            binding = configured_model.create_bindings(output_buffers=output_buffers)
            binding.input().set_buffer(np.array(frame))
            return binding

        return [_frame_binding(frame) for frame in input_batch]

    def _output_data_type2dict(self, data_type: Optional[str]) -> Dict[str, str]:
        valid_types = {"float32", "uint8", "uint16"}
        data_type_dict: Dict[str, str] = {}

        for output_info in self.hef.get_output_vstream_infos():
            name = output_info.name
            if data_type is None:
                hef_type = str(output_info.format.type).split(".")[-1]
                data_type_dict[name] = hef_type
            else:
                if data_type.lower() not in valid_types:
                    raise ValueError(f"Invalid data_type: {data_type}. Must be one of {valid_types}")
                data_type_dict[name] = data_type

        return data_type_dict

    def close(self) -> None:
        if self.last_infer_job is not None:
            self.last_infer_job.wait(10000)
        if self.config_ctx:
            self.config_ctx.__exit__(None, None, None)


def create_shared_vdevice(
    scheduler_algorithm: HailoSchedulingAlgorithm = HailoSchedulingAlgorithm.ROUND_ROBIN,
    group_id: str = "SHARED",
) -> VDevice:
    params = VDevice.create_params()
    params.scheduling_algorithm = scheduler_algorithm
    params.group_id = group_id
    return VDevice(params)
