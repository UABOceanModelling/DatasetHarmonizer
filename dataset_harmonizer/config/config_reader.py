"""Harmonizer configuration: YAML loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigError(ValueError):
    """Raised when the configuration YAML file is malformed or incomplete."""


def _is_str_mapping(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(k, str) for k in value)


def _section(data: dict, key: str, required: bool = False) -> dict:
    value = data.get(key)
    if value is None:
        if required:
            raise ConfigError(f"missing required section '{key}'")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"section '{key}' must be a mapping")
    return value


def _build(cls, section: str, values: dict):
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown keys in section '{section}': {unknown} (allowed: {sorted(allowed)})"
        )
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid section '{section}': {exc}") from exc


@dataclass
class InputData:
    data_paths: Dict[str, str]

    def __post_init__(self):
        if not _is_str_mapping(self.data_paths) or not self.data_paths:
            raise ConfigError("'input_data.data_paths' must map dataset names to paths/glob patterns")
        if not all(isinstance(p, str) for p in self.data_paths.values()):
            raise ConfigError("'input_data.data_paths' values must be strings")


@dataclass
class VariableConfig:
    source_var: str
    dims_mapping: dict

    def __post_init__(self):
        if not _is_str_mapping(self.dims_mapping) or not self.dims_mapping:
            raise ConfigError("'dims_mapping' must be a non-empty mapping of source -> target dimension")


@dataclass
class Interpolation:
    target_grid: str
    target_reference_variable: Optional[str] = None
    variables: Dict[str, VariableConfig] = field(default_factory=dict)

    def __post_init__(self):
        self.variables = self.variables or {}
        if self.variables and not self.target_reference_variable:
            raise ConfigError(
                "'interpolation.target_reference_variable' is required when variables are interpolated"
            )


@dataclass
class AttributesToAdd:
    file: dict = field(default_factory=dict)
    variables: dict = field(default_factory=dict)

    def __post_init__(self):
        self.file = self.file or {}
        self.variables = self.variables or {}
        if not isinstance(self.file, dict) or not all(
            isinstance(v, dict) for v in self.variables.values()
        ):
            raise ConfigError(
                "'attributes_to_add' must look like {file: {attr: value}, variables: {var: {attr: value}}}"
            )


@dataclass
class Slicing:
    region_of_interest: dict = field(default_factory=dict)

    def __post_init__(self):
        self.region_of_interest = self.region_of_interest or {}
        for dim, bounds in self.region_of_interest.items():
            if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
                raise ConfigError(f"'slicing.region_of_interest.{dim}' must be [start, end]")


@dataclass
class Keep:
    """Variables/coordinates to keep. `None` means "keep everything"."""

    variables: Optional[List[str]] = None
    coords: Optional[List[str]] = None


@dataclass
class Transformations:
    interpolation: Interpolation
    combination: dict = field(default_factory=dict)
    renaming: dict = field(default_factory=dict)
    attributes_to_add: AttributesToAdd = field(default_factory=AttributesToAdd)
    coords_to_add: dict = field(default_factory=dict)
    slicing: Slicing = field(default_factory=Slicing)
    keep: Keep = field(default_factory=Keep)

    def __post_init__(self):
        self.combination = self.combination or {}
        self.renaming = self.renaming or {}
        self.coords_to_add = self.coords_to_add or {}
        for name, spec in self.combination.items():
            if not isinstance(spec, dict) or not isinstance(spec.get("variables"), dict):
                raise ConfigError(
                    f"'combination.{name}' must contain a 'variables' mapping"
                )
        if not _is_str_mapping(self.renaming):
            raise ConfigError("'renaming' must be a mapping of old name -> new name")


@dataclass
class Reader:
    combine: str = "by_coords"
    data_vars: str = "all"


@dataclass
class XarrayParameters:
    engine: str = "netcdf4"
    reader: Reader = field(default_factory=Reader)


@dataclass
class Output:
    directory: str = "output/"
    file_name: str = "harmonized.nc"


@dataclass
class ConfigReader:
    input_data: InputData
    transformations: Transformations
    xarray_parameters: XarrayParameters
    output: Output

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConfigReader":
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a YAML mapping at the top level")

        try:
            return cls(
                input_data=_build(InputData, "input_data", _section(data, "input_data", True)),
                transformations=cls._build_transformations(_section(data, "transformations", True)),
                xarray_parameters=cls._build_xarray(_section(data, "xarray")),
                output=_build(Output, "output", _section(data, "output")),
            )
        except ConfigError as exc:
            raise ConfigError(f"{path}: {exc}") from exc

    @staticmethod
    def _build_transformations(t: dict) -> Transformations:
        unknown = sorted(set(t) - {f.name for f in fields(Transformations)})
        if unknown:
            raise ConfigError(f"unknown keys in section 'transformations': {unknown}")

        interp = _section(t, "interpolation", required=True)
        variables = {}
        for name, spec in (interp.get("variables") or {}).items():
            if not isinstance(spec, dict):
                raise ConfigError(f"'interpolation.variables.{name}' must be a mapping")
            variables[name] = _build(VariableConfig, f"interpolation.variables.{name}", spec)
        interpolation = _build(
            Interpolation, "interpolation", {**interp, "variables": variables}
        )

        return Transformations(
            interpolation=interpolation,
            combination=t.get("combination"),
            renaming=t.get("renaming"),
            attributes_to_add=_build(
                AttributesToAdd, "attributes_to_add", _section(t, "attributes_to_add")
            ),
            coords_to_add=t.get("coords_to_add"),
            slicing=_build(Slicing, "slicing", _section(t, "slicing")),
            keep=_build(Keep, "keep", _section(t, "keep")),
        )

    @staticmethod
    def _build_xarray(x: dict) -> XarrayParameters:
        unknown = sorted(set(x) - {"engine", "reader"})
        if unknown:
            raise ConfigError(f"unknown keys in section 'xarray': {unknown}")
        values = {}
        if "engine" in x:
            values["engine"] = x["engine"]
        if "reader" in x:
            values["reader"] = _build(Reader, "xarray.reader", x["reader"] or {})
        return XarrayParameters(**values)
