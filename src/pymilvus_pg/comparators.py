"""高性能数据比较器模块

本模块提供了专门针对Milvus schema优化的高性能数据比较功能，
用于替换DeepDiff以提升性能并支持精度容忍。
"""

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ToleranceConfig:
    """数值比较容差配置类"""

    # 浮点数比较容差
    float_absolute_tolerance: float = 1e-6  # 绝对容差
    float_relative_tolerance: float = 1e-5  # 相对容差

    # 向量比较配置
    vector_absolute_tolerance: float = 1e-3  # 向量元素绝对容差
    vector_relative_tolerance: float = 1e-2  # 向量元素相对容差
    vector_sample_ratio: float = 1.0  # 向量采样比例 (0.0-1.0)
    vector_min_sample_size: int = 8  # 最小采样数量

    # 字符串比较配置
    string_case_sensitive: bool = True  # 字符串大小写敏感

    # JSON比较配置
    json_ignore_order: bool = True  # JSON数组忽略顺序
    json_strict_type: bool = False  # JSON严格类型检查

    # Schema感知配置
    handle_default_values: bool = True  # 处理默认值语义
    handle_nullable_fields: bool = True  # 处理可空字段语义
    handle_dynamic_fields: bool = True  # 处理动态字段（$meta）
    
    # 默认值处理策略
    treat_missing_as_default: bool = True  # 将缺失值视为默认值
    ignore_default_value_diffs: bool = True  # 忽略默认值差异
    
    # Null值处理策略
    null_equals_missing: bool = True  # null等同于缺失值
    null_equals_default: bool = False  # null等同于默认值


class FieldComparator(ABC):
    """字段比较器抽象基类"""

    def __init__(self, field_name: str, tolerance_config: ToleranceConfig):
        self.field_name = field_name
        self.tolerance = tolerance_config

    @abstractmethod
    def compare(self, value1: Any, value2: Any) -> bool:
        """比较两个值是否相等

        Returns:
            bool: True表示相等，False表示不相等
        """
        pass


class NumericComparator(FieldComparator):
    """数值字段比较器，支持容差比较"""

    def __init__(self, field_name: str, tolerance_config: ToleranceConfig, data_type: str):
        super().__init__(field_name, tolerance_config)
        self.data_type = data_type
        self.is_integer = data_type in ["INT8", "INT16", "INT32", "INT64"]

    def compare(self, value1: Any, value2: Any) -> bool:
        """数值比较，整数精确比较，浮点数容差比较"""
        if value1 is None and value2 is None:
            return True
        if value1 is None or value2 is None:
            return False

        try:
            # 转换为数值类型
            num1 = float(value1) if not self.is_integer else int(value1)
            num2 = float(value2) if not self.is_integer else int(value2)

            if self.is_integer:
                return num1 == num2

            # 浮点数容差比较
            return math.isclose(
                num1,
                num2,
                abs_tol=self.tolerance.float_absolute_tolerance,
                rel_tol=self.tolerance.float_relative_tolerance,
            )
        except (ValueError, TypeError):
            # 无法转换为数值时，回退到字符串比较
            return str(value1) == str(value2)


class StringComparator(FieldComparator):
    """字符串字段比较器"""

    def compare(self, value1: Any, value2: Any) -> bool:
        """字符串比较"""
        if value1 is None and value2 is None:
            return True
        if value1 is None or value2 is None:
            return False

        str1 = str(value1)
        str2 = str(value2)

        if not self.tolerance.string_case_sensitive:
            str1 = str1.lower()
            str2 = str2.lower()

        return str1 == str2


class VectorComparator(FieldComparator):
    """向量字段高性能比较器，使用NumPy优化"""

    def __init__(self, field_name: str, tolerance_config: ToleranceConfig):
        super().__init__(field_name, tolerance_config)

    def _sample_vector(self, vector: list[Any] | np.ndarray) -> np.ndarray:
        """向量采样，减少比较开销"""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        vector_len = len(vector)
        sample_ratio = self.tolerance.vector_sample_ratio

        if sample_ratio >= 1.0 or vector_len <= self.tolerance.vector_min_sample_size:
            return vector

        # 计算采样数量
        sample_size = max(self.tolerance.vector_min_sample_size, int(vector_len * sample_ratio))

        if sample_size >= vector_len:
            return vector

        # 均匀采样：包含首尾元素
        indices = np.linspace(0, vector_len - 1, sample_size, dtype=int)
        return np.array(vector[indices], dtype=np.float32)

    def compare(self, value1: Any, value2: Any) -> bool:
        """向量比较，使用NumPy向量化操作"""
        if value1 is None and value2 is None:
            return True
        if value1 is None or value2 is None:
            return False

        try:
            # 转换为NumPy数组
            vec1 = self._sample_vector(value1)
            vec2 = self._sample_vector(value2)

            # 长度检查
            if len(vec1) != len(vec2):
                return False

            # 使用NumPy的向量化容差比较
            return np.allclose(
                vec1, vec2, atol=self.tolerance.vector_absolute_tolerance, rtol=self.tolerance.vector_relative_tolerance
            )
        except (ValueError, TypeError):
            # 回退到字符串比较
            return str(value1) == str(value2)


class ArrayComparator(FieldComparator):
    """数组字段批量比较器"""

    def __init__(self, field_name: str, tolerance_config: ToleranceConfig, element_type: str):
        super().__init__(field_name, tolerance_config)
        self.element_type = element_type
        # 为数组元素创建相应的比较器
        if element_type in ["FLOAT", "DOUBLE"]:
            self.element_comparator: FieldComparator = NumericComparator(
                f"{field_name}_element", tolerance_config, element_type
            )
        else:
            self.element_comparator = StringComparator(f"{field_name}_element", tolerance_config)

    def compare(self, value1: Any, value2: Any) -> bool:
        """数组比较，逐元素比较"""
        if value1 is None and value2 is None:
            return True
        if value1 is None or value2 is None:
            return False

        try:
            list1 = list(value1) if not isinstance(value1, list) else value1
            list2 = list(value2) if not isinstance(value2, list) else value2

            if len(list1) != len(list2):
                return False

            # 逐元素比较
            for elem1, elem2 in zip(list1, list2, strict=False):
                if not self.element_comparator.compare(elem1, elem2):
                    return False

            return True
        except (ValueError, TypeError):
            return str(value1) == str(value2)


class JSONComparator(FieldComparator):
    """JSON字段结构化比较器"""

    def _normalize_json(self, value: Any) -> Any:
        """JSON值标准化"""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def _compare_json_recursive(self, obj1: Any, obj2: Any) -> bool:
        """递归比较JSON对象"""
        if type(obj1) is not type(obj2):
            if not self.tolerance.json_strict_type:
                # 尝试类型转换
                try:
                    if isinstance(obj1, (int, float)) and isinstance(obj2, (int, float)):
                        return math.isclose(
                            float(obj1),
                            float(obj2),
                            abs_tol=self.tolerance.float_absolute_tolerance,
                            rel_tol=self.tolerance.float_relative_tolerance,
                        )
                except (ValueError, TypeError):
                    pass
            return False

        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            return all(self._compare_json_recursive(obj1[key], obj2[key]) for key in obj1.keys())

        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                return False

            if self.tolerance.json_ignore_order:
                # 忽略顺序的比较（适用于简单类型）
                try:
                    if all(not isinstance(x, (dict, list)) for x in obj1 + obj2):
                        return sorted(str(x) for x in obj1) == sorted(str(x) for x in obj2)
                except Exception:
                    pass

            # 有序比较
            return all(self._compare_json_recursive(elem1, elem2) for elem1, elem2 in zip(obj1, obj2, strict=False))

        elif isinstance(obj1, (int, float)):
            return math.isclose(
                float(obj1),
                float(obj2),
                abs_tol=self.tolerance.float_absolute_tolerance,
                rel_tol=self.tolerance.float_relative_tolerance,
            )

        else:
            return bool(obj1 == obj2)

    def compare(self, value1: Any, value2: Any) -> bool:
        """JSON比较"""
        if value1 is None and value2 is None:
            return True
        if value1 is None or value2 is None:
            return False

        try:
            norm1 = self._normalize_json(value1)
            norm2 = self._normalize_json(value2)
            return self._compare_json_recursive(norm1, norm2)
        except Exception:
            # 回退到字符串比较
            return str(value1) == str(value2)


class BooleanComparator(FieldComparator):
    """布尔字段比较器"""

    def compare(self, value1: Any, value2: Any) -> bool:
        """布尔值比较"""
        if value1 is None and value2 is None:
            return True
        if value1 is None or value2 is None:
            return False

        try:
            return bool(value1) == bool(value2)
        except (ValueError, TypeError):
            return str(value1) == str(value2)


class DefaultValueAwareComparator:
    """默认值感知比较器包装器"""
    
    def __init__(
        self, 
        base_comparator: FieldComparator, 
        field_schema: dict[str, Any],
        tolerance_config: ToleranceConfig
    ):
        self.base_comparator = base_comparator
        self.field_schema = field_schema
        self.tolerance = tolerance_config
        self.field_name = field_schema.get('name', 'unknown')
        self.default_value = field_schema.get('default_value')
        self.is_nullable = field_schema.get('nullable', False)
    
    def _normalize_value(self, value: Any) -> Any:
        """标准化值，处理默认值和null语义"""
        # 处理None值
        if value is None:
            if self.tolerance.null_equals_default and self.default_value is not None:
                return self.default_value
            return None
        
        # 处理缺失值（当treat_missing_as_default为True时）
        if (self.tolerance.treat_missing_as_default and 
            self.default_value is not None and 
            value == self.default_value):
            return self.default_value
            
        return value
    
    def compare(self, value1: Any, value2: Any) -> bool:
        """Schema感知的值比较"""
        norm1 = self._normalize_value(value1)
        norm2 = self._normalize_value(value2)
        
        # 如果忽略默认值差异且有默认值
        if (self.tolerance.ignore_default_value_diffs and 
            self.default_value is not None):
            # 检查是否一个是默认值，一个是None
            if ((norm1 == self.default_value and norm2 is None) or
                (norm1 is None and norm2 == self.default_value)):
                return True
            # 检查是否都是默认值
            if norm1 == self.default_value and norm2 == self.default_value:
                return True
        
        # 如果两个值都是None
        if norm1 is None and norm2 is None:
            return True
        
        # 如果一个是None，另一个不是
        if norm1 is None or norm2 is None:
            # 对于nullable字段，考虑null语义
            if self.is_nullable:
                if self.tolerance.null_equals_missing:
                    return False  # 只有当两个都是None时才相等
                if self.tolerance.null_equals_default and self.default_value is not None:
                    # None等同于默认值
                    non_null_value = norm1 if norm2 is None else norm2
                    return non_null_value == self.default_value
            return False
        
        # 使用基础比较器进行实际比较
        return self.base_comparator.compare(norm1, norm2)


class DynamicFieldComparator:
    """动态字段($meta)比较器"""
    
    def __init__(self, tolerance_config: ToleranceConfig):
        self.tolerance = tolerance_config
        self.json_comparator = JSONComparator("$meta", tolerance_config)
    
    def compare_dynamic_fields(
        self, 
        meta1: dict[str, Any] | None, 
        meta2: dict[str, Any] | None
    ) -> tuple[bool, list[str]]:
        """比较动态字段内容
        
        Args:
            meta1: 第一个记录的$meta字段
            meta2: 第二个记录的$meta字段
            
        Returns:
            Tuple[bool, List[str]]: (是否相等, 差异字段名列表)
        """
        if not self.tolerance.handle_dynamic_fields:
            return True, []  # 如果禁用动态字段处理，直接返回相等
        
        if meta1 is None and meta2 is None:
            return True, []
        
        if meta1 is None or meta2 is None:
            # 一个有动态字段，一个没有
            diff_fields = list((meta1 or meta2).keys())
            return False, diff_fields
        
        # 比较动态字段内容
        all_dynamic_keys = set(meta1.keys()) | set(meta2.keys())
        differences = []
        
        for key in all_dynamic_keys:
            val1 = meta1.get(key)
            val2 = meta2.get(key)
            
            if not self.json_comparator.compare(val1, val2):
                differences.append(f"$meta.{key}")
        
        return len(differences) == 0, differences


class HighPerformanceComparator:
    """高性能数据比较器主类"""

    def __init__(
        self,
        field_types: dict[str, str],
        field_categories: dict[str, list[str]],
        tolerance_config: ToleranceConfig | None = None,
        field_schemas: dict[str, dict[str, Any]] | None = None,  # 新增：字段schema信息
    ):
        """
        初始化高性能比较器

        Args:
            field_types: 字段名到数据类型的映射
            field_categories: 字段分类信息 (json_fields, array_fields等)
            tolerance_config: 容差配置
            field_schemas: 字段schema详细信息（包含nullable、default_value等）
        """
        self.tolerance = tolerance_config or ToleranceConfig()
        self.field_types = field_types
        self.field_categories = field_categories
        self.field_schemas = field_schemas or {}

        # 预编译字段比较器
        self.field_comparators = self._build_field_comparators()
        
        # 动态字段比较器
        self.dynamic_comparator = DynamicFieldComparator(self.tolerance)

    def _build_field_comparators(self) -> dict[str, FieldComparator | DefaultValueAwareComparator]:
        """基于schema预编译字段比较器"""
        comparators: dict[str, FieldComparator | DefaultValueAwareComparator] = {}

        for field_name, data_type in self.field_types.items():
            # 创建基础比较器
            base_comparator: FieldComparator
            
            if field_name in self.field_categories.get("json_fields", []):
                base_comparator = JSONComparator(field_name, self.tolerance)

            elif field_name in self.field_categories.get("float_vector_fields", []):
                base_comparator = VectorComparator(field_name, self.tolerance)

            elif field_name in self.field_categories.get("array_fields", []):
                # 尝试从schema获取数组元素类型
                field_schema = self.field_schemas.get(field_name, {})
                element_type = field_schema.get('element_type', 'VARCHAR')
                base_comparator = ArrayComparator(field_name, self.tolerance, element_type)

            elif data_type in ["FLOAT", "DOUBLE", "INT8", "INT16", "INT32", "INT64"]:
                base_comparator = NumericComparator(field_name, self.tolerance, data_type)

            elif data_type == "BOOL":
                base_comparator = BooleanComparator(field_name, self.tolerance)

            else:  # VARCHAR 和其他类型
                base_comparator = StringComparator(field_name, self.tolerance)
            
            # 如果启用schema感知处理且有字段schema信息，使用DefaultValueAwareComparator包装
            field_schema = self.field_schemas.get(field_name)
            if (self.tolerance.handle_default_values or self.tolerance.handle_nullable_fields) and field_schema:
                has_default = 'default_value' in field_schema
                is_nullable = field_schema.get('nullable', False)
                
                if has_default or is_nullable:
                    comparators[field_name] = DefaultValueAwareComparator(
                        base_comparator, field_schema, self.tolerance
                    )
                else:
                    comparators[field_name] = base_comparator
            else:
                comparators[field_name] = base_comparator
        
        return comparators

    def compare_records(
        self, record1: dict[str, Any], record2: dict[str, Any], fields_to_compare: set | None = None
    ) -> tuple[bool, list[str]]:
        """
        比较两条记录，支持动态字段和schema感知

        Args:
            record1: 第一条记录
            record2: 第二条记录
            fields_to_compare: 要比较的字段集合，None表示比较所有字段

        Returns:
            Tuple[bool, List[str]]: (是否相等, 差异字段列表)
        """
        if fields_to_compare is None:
            fields_to_compare = set(record1.keys()) & set(record2.keys())

        differences = []

        # 比较常规字段
        for field_name in fields_to_compare:
            # 跳过动态字段，单独处理
            if field_name == '$meta':
                continue
                
            if field_name not in self.field_comparators:
                # 回退到简单比较
                if record1.get(field_name) != record2.get(field_name):
                    differences.append(field_name)
                continue

            comparator = self.field_comparators[field_name]
            if not comparator.compare(record1.get(field_name), record2.get(field_name)):
                differences.append(field_name)

        # 处理动态字段（$meta）
        if self.tolerance.handle_dynamic_fields:
            meta1 = record1.get('$meta')
            meta2 = record2.get('$meta')
            
            is_meta_equal, meta_differences = self.dynamic_comparator.compare_dynamic_fields(meta1, meta2)
            if not is_meta_equal:
                differences.extend(meta_differences)

        return len(differences) == 0, differences

    def compare_dataframes(
        self, df1: pd.DataFrame, df2: pd.DataFrame, primary_key: str
    ) -> tuple[bool, list[dict[str, Any]]]:
        """
        批量比较两个DataFrame

        Args:
            df1: 第一个DataFrame
            df2: 第二个DataFrame
            primary_key: 主键字段名

        Returns:
            Tuple[bool, List[Dict]]: (是否完全相等, 差异详情列表)
        """
        differences = []

        # 对齐DataFrame
        common_index = df1.index.intersection(df2.index)
        aligned_df1 = df1.loc[common_index].sort_index()
        aligned_df2 = df2.loc[common_index].sort_index()

        common_columns = set(aligned_df1.columns) & set(aligned_df2.columns)

        # 逐行比较
        for pk in common_index:
            record1 = aligned_df1.loc[pk].to_dict()
            record2 = aligned_df2.loc[pk].to_dict()

            is_equal, diff_fields = self.compare_records(record1, record2, common_columns)

            if not is_equal:
                differences.append(
                    {
                        "primary_key": pk,
                        "different_fields": diff_fields,
                        "milvus_values": {field: record1.get(field) for field in diff_fields},
                        "pg_values": {field: record2.get(field) for field in diff_fields},
                    }
                )

        return len(differences) == 0, differences


def create_comparator_from_schema(
    schema_info: dict[str, Any], tolerance_config: ToleranceConfig | None = None
) -> HighPerformanceComparator:
    """
    从schema信息创建比较器

    Args:
        schema_info: schema信息字典，包含字段类型、分类和详细schema
        tolerance_config: 容差配置

    Returns:
        HighPerformanceComparator: 配置好的比较器实例
    """
    return HighPerformanceComparator(
        field_types=schema_info.get("field_types", {}),
        field_categories=schema_info.get("field_categories", {}),
        tolerance_config=tolerance_config,
        field_schemas=schema_info.get("field_schemas", {}),  # 新增：支持详细schema信息
    )


def create_schema_aware_comparator(
    field_types: dict[str, str],
    field_categories: dict[str, list[str]], 
    field_schemas: dict[str, dict[str, Any]],
    tolerance_config: ToleranceConfig | None = None
) -> HighPerformanceComparator:
    """
    创建schema感知的比较器（便捷函数）

    Args:
        field_types: 字段名到数据类型的映射
        field_categories: 字段分类信息
        field_schemas: 字段详细schema信息
        tolerance_config: 容差配置

    Returns:
        HighPerformanceComparator: 配置好的schema感知比较器实例
    """
    if tolerance_config is None:
        tolerance_config = ToleranceConfig(
            handle_default_values=True,
            handle_nullable_fields=True,
            handle_dynamic_fields=True
        )
    
    return HighPerformanceComparator(
        field_types=field_types,
        field_categories=field_categories,
        tolerance_config=tolerance_config,
        field_schemas=field_schemas,
    )
