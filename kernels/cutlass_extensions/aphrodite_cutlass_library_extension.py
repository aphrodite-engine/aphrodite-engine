import enum
from typing import Union

from cutlass_library import *

#
#   Extend cutlass library with custom types, and missing values
#


class APHRODITEDataType(enum.Enum):
    u4b8 = enum_auto()
    u8b128 = enum_auto()


class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecialized = enum_auto()
    TmaWarpSpecializedPingpong = enum_auto()
    TmaWarpSpecializedCooperative = enum_auto()


APHRODITEDataTypeNames: dict[Union[APHRODITEDataType, DataType], str] = {
    **DataTypeNames,  # type: ignore
    **{
        APHRODITEDataType.u4b8: "u4b8",
        APHRODITEDataType.u8b128: "u8b128",
    }
}

APHRODITEDataTypeTag: dict[Union[APHRODITEDataType, DataType], str] = {
    **DataTypeTag,  # type: ignore
    **{
        APHRODITEDataType.u4b8: "cutlass::aphrodite_uint4b8_t",
        APHRODITEDataType.u8b128: "cutlass::aphrodite_uint8b128_t",
    }
}

APHRODITEDataTypeSize: dict[Union[APHRODITEDataType, DataType], int] = {
    **DataTypeSize,  # type: ignore
    **{
        APHRODITEDataType.u4b8: 4,
        APHRODITEDataType.u8b128: 8,
    }
}

APHRODITEDataTypeAPHRODITEScalarTypeTag: dict[Union[APHRODITEDataType,
                                                    DataType],
                                              str] = {
                                                  APHRODITEDataType.u4b8:
                                                  "aphrodite::kU4B8",
                                                  APHRODITEDataType.u8b128:
                                                  "aphrodite::kU8B128",
                                                  DataType.u4:
                                                  "aphrodite::kU4",
                                                  DataType.u8:
                                                  "aphrodite::kU8",
                                                  DataType.s4:
                                                  "aphrodite::kS4",
                                                  DataType.s8:
                                                  "aphrodite::kS8",
                                                  DataType.f16:
                                                  "aphrodite::kFloat16",
                                                  DataType.bf16:
                                                  "aphrodite::kBfloat16",
                                              }

APHRODITEDataTypeTorchDataTypeTag: dict[Union[APHRODITEDataType, DataType],
                                        str] = {
                                            DataType.u8:
                                            "at::ScalarType::Byte",
                                            DataType.s8:
                                            "at::ScalarType::Char",
                                            DataType.e4m3:
                                            "at::ScalarType::Float8_e4m3fn",
                                            DataType.s32:
                                            "at::ScalarType::Int",
                                            DataType.f16:
                                            "at::ScalarType::Half",
                                            DataType.bf16:
                                            "at::ScalarType::BFloat16",
                                            DataType.f32:
                                            "at::ScalarType::Float",
                                        }

APHRODITEKernelScheduleTag: dict[Union[
    MixedInputKernelScheduleType, KernelScheduleType], str] = {
        **KernelScheduleTag,  # type: ignore
        **{
            MixedInputKernelScheduleType.TmaWarpSpecialized:
            "cutlass::gemm::KernelTmaWarpSpecialized",
            MixedInputKernelScheduleType.TmaWarpSpecializedPingpong:
            "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
            MixedInputKernelScheduleType.TmaWarpSpecializedCooperative:
            "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
        }
    }
