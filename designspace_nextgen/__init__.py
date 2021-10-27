from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import fontTools.misc.plistlib
import fontTools.varLib.models  # type: ignore
from fontTools.misc import etree as ElementTree  # type: ignore

__version__ = "0.1.0"

LOGGER = logging.getLogger(__name__)

Location = dict[str, Union[float, tuple[float, float]]]
# IsotropicLocation from axes?

# TODO: Remove after fontTools.varLib.models is typed.
PiecewiseRemap = Callable[[float, Mapping[float, float]], float]

# TODO: label_names, localised_... -> use to store _all_ names and provide properties
# for the default "en" ones? Simplifies writing?


class Error(Exception):
    """Base exception."""


@dataclass(frozen=True)
class Document:
    axes: list[Axis]
    rules: list[Rule] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    instances: list[Instance] = field(default_factory=list)
    path: Optional[Path] = None
    format_version: float = 4
    format_version_minor: float = 1
    rules_processing_last: bool = field(default=False)
    lib: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.axes:
            raise Error(f"A Designspace must have at least one axis.")
        source_filenames: set[Path] = set(
            s.filename for s in self.sources if s.filename is not None
        )
        for instance in self.instances:
            if instance.filename is not None and instance.filename in source_filenames:
                raise Error(
                    f"Instance '{instance.name}' has a file name identical to a source: {instance.filename}"
                )

    @classmethod
    def from_bytes(cls, content: bytes, path: Optional[Path] = None) -> Document:
        root = ElementTree.fromstring(content)

        axes, default_location = _read_axes(root)
        rules, rules_processing_last = _read_rules(root)
        sources = _read_sources(root, default_location)
        instances = _read_instances(root, default_location)
        lib = _read_lib(root)

        return cls(
            axes=axes,
            rules=rules,
            rules_processing_last=rules_processing_last,
            sources=sources,
            instances=instances,
            lib=lib,
            path=path,
        )

    @classmethod
    def from_file(cls, path: os.PathLike[str]) -> Document:
        path = Path(path)
        try:
            document = cls.from_bytes(path.read_bytes(), path)
        except Exception as e:
            raise Error(f"Failed to read Designspace from '{path}': {str(e)}") from e

        return document

    def save(self, path: Optional[os.PathLike[str]] = None) -> None:
        if path is None:
            if self.path is None:
                raise Error("Document has no known path and no path was given.")
            path = self.path
        else:
            path = Path(path)

        root = ElementTree.Element("designspace")
        root.attrib["format"] = f"{self.format_version}.{self.format_version_minor}"

        try:
            _write_axes(self.axes, root)
            _write_rules(self.rules, self.rules_processing_last, root)
            default_location: Location = self.default_design_location()
            _write_sources(self.sources, default_location, root)
            _write_instances(self.instances, default_location, root)
            _write_lib(self.lib, root)

            tree = ElementTree.ElementTree(root)
            tree.write(
                os.fspath(path),
                encoding="UTF-8",
                method="xml",
                xml_declaration=True,
                pretty_print=True,
            )
        except Exception as e:
            raise Error(f"Failed to write Designspace to path {path}: {str(e)}")

    def default_design_location(self) -> Location:
        return {axis.name: axis.map_forward(axis.default) for axis in self.axes}

    def default_source(self) -> Optional[Source]:
        default_location = self.default_design_location()
        default_sources = [s for s in self.sources if s.location == default_location]
        if not default_sources:
            return None
        elif len(default_sources) == 1:
            return default_sources[0]
        raise Error(
            f"More than one default source found at location {default_location}: "
            f"{', '.join(s.name for s in default_sources)}"
        )

    def evaluate_rules(
        self, location: Location, glyph_names: set[str]
    ) -> list[tuple[str, str]]:
        """Applies substitution rules to glyphs at location, returning a list of
        old to new name tuples, in the order of the rules.

        Raises an error if a substitution references a glyph not in `glyph_names`.
        """

        swaps: list[tuple[str, str]] = []
        for rule in self.rules:
            swaps.extend(rule.evaluate(location, glyph_names))
        return swaps


@dataclass(frozen=True)
class Axis:
    name: str  # name of the axis used in locations
    minimum: float
    default: float
    maximum: float
    tag: Optional[str]  # opentype tag for this axis
    label_names: dict[str, str] = field(default_factory=dict)
    hidden: bool = False
    mapping: dict[float, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tag is not None and len(self.tag) != 4:
            raise Error(f"An axis tag must consist of 4 characters.")

    def map_forward(self, value: float) -> float:
        if self.mapping:
            remap: PiecewiseRemap = fontTools.varLib.models.piecewiseLinearMap
            return remap(value, self.mapping)
        return value

    def map_backward(self, value: float) -> float:
        if self.mapping:
            remap: PiecewiseRemap = fontTools.varLib.models.piecewiseLinearMap
            return remap(value, {v: k for k, v in self.mapping.items()})
        return value


@dataclass(frozen=True)
class Source:
    name: str
    filename: Optional[Path] = None
    location: Location = field(default_factory=dict)
    font: Optional[Any] = None
    layer_name: Optional[str] = None
    family_name: Optional[str] = None
    style_name: Optional[str] = None


@dataclass(frozen=True)
class Instance:
    name: str
    filename: Optional[Path] = None
    location: Location = field(default_factory=dict)
    font: Optional[Any] = None
    family_name: Optional[str] = None
    style_name: Optional[str] = None
    postscript_font_name: Optional[str] = None
    style_map_family_name: Optional[str] = None
    style_map_style_name: Optional[str] = None
    localised_style_name: dict[str, str] = field(default_factory=dict)
    localised_family_name: dict[str, str] = field(default_factory=dict)
    localised_style_map_style_name: dict[str, str] = field(default_factory=dict)
    localised_style_map_family_name: dict[str, str] = field(default_factory=dict)
    lib: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Rule:
    name: str
    condition_sets: list["ConditionSet"]
    substitutions: dict[str, str]

    def __post_init__(self) -> None:
        if not self.condition_sets:
            raise Error(f"Rule '{self.name}': Must have at least one condition set.")
        if not self.substitutions:
            raise Error(f"Rule '{self.name}': Must have at least one substitution.")

    def applies_to(self, location: Location) -> bool:
        """Returns true if any condition set applies to the location, false otherwise."""

        return any(c.applies_to(location) for c in self.condition_sets)

    def evaluate(
        self, location: Location, glyph_names: set[str]
    ) -> list[tuple[str, str]]:
        """Applies substitution rules to glyphs at location, returning a list of
        old to new name tuples.

        Raises an error if a substitution references a glyph not in `glyph_names`.
        """

        swaps: list[tuple[str, str]] = []
        if not self.applies_to(location):
            return swaps
        for old_name, new_name in self.substitutions.items():
            if old_name not in glyph_names:
                raise Error(
                    f"Rule '{self.name}' references glyph '{old_name}' "
                    "which does not exist in the glyph set."
                )
            if new_name not in glyph_names:
                raise Error(
                    f"Rule '{self.name}' references glyph '{new_name}' "
                    "which does not exist in the glyph set."
                )
            swaps.append((old_name, new_name))
        return swaps


@dataclass(frozen=True)
class ConditionSet:
    conditions: list["Condition"]

    def applies_to(self, location: Location) -> bool:
        """Returns true if all conditions in the set apply to the location, false otherwise.

        NOTE: An empty condition set always applies.
        """

        return all(c.applies_to(location) for c in self.conditions)


@dataclass(frozen=True)
class Condition:
    name: str  # Axis name the condition applies to.
    minimum: float = -math.inf
    maximum: float = math.inf

    def __post_init__(self) -> None:
        if self.minimum == -math.inf and self.maximum == math.inf:
            raise Error(
                f"Condition '{self.name}': either minimum, maximum or both must be set."
            )

    def applies_to(self, location: Location) -> bool:
        """Returns true if the condition applies to the location, false otherwise."""

        value = location.get(self.name)
        if value is None:
            return False
        if isinstance(value, tuple):
            raise Error(f"Cannot evaluate rules for anisotropic locations: {value}")
        return self.minimum <= value <= self.maximum


###

# ElementTree allows to find namespace-prefixed elements, but not attributes
# so we have to do it ourselves for 'xml:lang'
XML_NS = "{http://www.w3.org/XML/1998/namespace}"
XML_LANG = XML_NS + "lang"


def _read_axes(tree: ElementTree.Element) -> tuple[list[Axis], Mapping[str, float]]:
    stray_map_element = tree.find(".axes/map")
    if stray_map_element:
        raise Error(
            "Stray <map> elements found in <axes> element. They must be subelements of "
            "the <axes> element."
        )

    axes: list[Axis] = []
    default_location: dict[str, float] = {}
    for index, element in enumerate(tree.findall(".axes/axis")):
        attributes = element.attrib

        name = attributes.get("name")
        if name is None:
            raise Error(f"Axis at index {index} needs a name.")
        tag = attributes.get("tag")
        minimum = attributes.get("minimum")
        if minimum is None:
            raise Error(f"Axis '{name}' needs a minimum value.")
        default = attributes.get("default")
        if default is None:
            raise Error(f"Axis '{name}' needs a default value.")
        maximum = attributes.get("maximum")
        if maximum is None:
            raise Error(f"Axis '{name}' needs a maximum value.")
        hidden = bool(attributes.get("hidden", False))
        mapping = {
            float(m.attrib["input"]): float(m.attrib["output"])
            for m in element.findall("map")
        }
        label_names = {
            lang: label_name.text or ""
            for label_name in element.findall("labelname")
            for key, lang in label_name.items()
            if key == XML_LANG
            # Note: elementtree reads the "xml:lang" attribute name as
            # '{http://www.w3.org/XML/1998/namespace}lang'
        }

        axis = Axis(
            name=name,
            tag=tag,
            minimum=float(minimum),
            default=float(default),
            maximum=float(maximum),
            hidden=hidden,
            mapping=mapping,
            label_names=label_names,
        )
        axes.append(axis)
        default_location[name] = axis.map_forward(axis.default)

    return axes, default_location


def _read_rules(tree: ElementTree.Element) -> tuple[list[Rule], bool]:
    rule_element = tree.find(".rules")
    rules_processing_last = False
    if rule_element is not None:
        processing = rule_element.attrib.get("processing", "first")
        if processing not in {"first", "last"}:
            raise Error(
                f"<rules> processing attribute value is not valid: {processing:r}, "
                "expected 'first' or 'last'."
            )
        rules_processing_last = processing == "last"

    rules: list[Rule] = []
    for index, element in enumerate(tree.findall(".rules/rule")):
        name = element.attrib.get("name")
        if name is None:
            raise Error(
                f"Rule at index {index} needs a name so I can properly error at you."
            )

        # read any stray conditions outside a condition set
        condition_sets: list[ConditionSet] = []
        conditions_external = _read_conditions(element, name)
        if conditions_external:
            condition_sets.append(ConditionSet(conditions_external))
        # read the conditionsets
        for conditionset_element in element.findall(".conditionset"):
            condition_sets.append(
                ConditionSet(_read_conditions(conditionset_element, name))
            )
        if not condition_sets:
            raise Error(f"Rule '{name}' needs at least one condition.")

        substitutions = {
            sub_element.attrib["name"]: sub_element.attrib["with"]
            for sub_element in element.findall(".sub")
        }

        rules.append(
            Rule(name=name, condition_sets=condition_sets, substitutions=substitutions)
        )

    return rules, rules_processing_last


def _read_conditions(parent: ElementTree.Element, rule_name: str) -> list[Condition]:
    conditions: list[Condition] = []

    for element in parent.findall(".condition"):
        attributes = element.attrib

        name = attributes.get("name")
        if name is None:
            raise Error(
                f"Rule '{rule_name}': Conditions must have names with the axis name they apply to."
            )
        minimum = attributes.get("minimum")
        maximum = attributes.get("maximum")
        if minimum is None and maximum is None:
            raise Error(
                f"Rule '{rule_name}': Conditions must have either a minimum, a maximum or both."
            )

        conditions.append(
            Condition(
                name=name,
                minimum=float(minimum) if minimum is not None else -math.inf,
                maximum=float(maximum) if maximum is not None else math.inf,
            )
        )

    return conditions


def _read_sources(
    tree: ElementTree.Element, default_location: Mapping[str, float]
) -> list[Source]:
    sources: list[Source] = []

    for index, element in enumerate(tree.findall(".sources/source")):
        attributes = element.attrib

        name = attributes.get("name")
        if name is None:
            name = f"temp_master.{index}"
        filename = attributes.get("filename")
        family_name = attributes.get("familyname")
        style_name = attributes.get("stylename")
        location = _read_location(element, default_location)
        layer_name = attributes.get("layer")

        sources.append(
            Source(
                name=name,
                filename=Path(filename) if filename is not None else None,
                location=location,
                layer_name=layer_name,
                family_name=family_name,
                style_name=style_name,
            )
        )

    return sources


def _read_location(
    element: ElementTree.Element, default_location: Mapping[str, float]
) -> Location:
    location: Location = {k: v for k, v in default_location.items()}
    location_element = element.find(".location")
    if location_element is None:
        return location  # Return copy of default location.

    for dimension_element in location_element.findall(".dimension"):
        attributes = dimension_element.attrib

        name = attributes.get("name")
        if name is None:
            raise Error("Locations must have a name.")
        if name not in default_location:
            LOGGER.warning('Location with unknown axis "%s", skipping.', name)
            continue

        x_value = attributes.get("xvalue")
        if x_value is None:
            raise Error(f"Location for axis '{name}' needs at least an xvalue.")
        x = float(x_value)

        y = None
        y_value = attributes.get("yvalue")
        if y_value is not None:
            y = float(y_value)

        if y is not None:
            location[name] = (x, y)
        else:
            location[name] = x

    return location


def _read_instances(
    tree: ElementTree.Element, default_location: Mapping[str, float]
) -> list[Instance]:
    instances: list[Instance] = []

    for index, instance_element in enumerate(tree.findall(".instances/instance")):
        attributes = instance_element.attrib

        name = attributes.get("name")
        if name is None:
            name = f"temp_instance.{index}"
        location = _read_location(instance_element, default_location)
        filename = attributes.get("filename")
        family_name = attributes.get("familyname")
        style_name = attributes.get("stylename")
        postscript_font_name = attributes.get("postscriptfontname")
        style_map_family_name = attributes.get("stylemapfamilyname")
        style_map_style_name = attributes.get("stylemapstylename")
        lib = _read_lib(instance_element)

        localised_style_name = {
            lang: element.text or ""
            for element in instance_element.findall("stylename")
            for key, lang in element.items()
            if key == XML_LANG
        }
        localised_family_name = {
            lang: element.text or ""
            for element in instance_element.findall("familyname")
            for key, lang in element.items()
            if key == XML_LANG
        }
        localised_style_map_style_name = {
            lang: element.text or ""
            for element in instance_element.findall("stylemapstylename")
            for key, lang in element.items()
            if key == XML_LANG
        }
        localised_style_map_family_name = {
            lang: element.text or ""
            for element in instance_element.findall("stylemapfamilyname")
            for key, lang in element.items()
            if key == XML_LANG
        }

        instances.append(
            Instance(
                name=name,
                filename=Path(filename) if filename is not None else None,
                location=location,
                family_name=family_name,
                style_name=style_name,
                postscript_font_name=postscript_font_name,
                style_map_family_name=style_map_family_name,
                style_map_style_name=style_map_style_name,
                localised_style_name=localised_style_name,
                localised_family_name=localised_family_name,
                localised_style_map_style_name=localised_style_map_style_name,
                localised_style_map_family_name=localised_style_map_family_name,
                lib=lib,
            )
        )

    return instances


def _read_lib(tree: ElementTree.Element) -> dict[str, Any]:
    lib_element = tree.find(".lib")
    if lib_element is not None:
        return fontTools.misc.plistlib.fromtree(lib_element)
    return {}


###


def int_or_float_to_str(num: float) -> str:
    return f"{num:f}".rstrip("0").rstrip(".")


def _write_axes(axes: list[Axis], root: ElementTree.Element) -> None:
    if not axes:
        raise Error("Designspace must have at least one axis.")

    axes_element = ElementTree.Element("axes")
    for axis in axes:
        axis_element = ElementTree.Element("axis")
        if axis.tag is not None:
            axis_element.attrib["tag"] = axis.tag
        axis_element.attrib["name"] = axis.name
        axis_element.attrib["minimum"] = int_or_float_to_str(axis.minimum)
        axis_element.attrib["maximum"] = int_or_float_to_str(axis.maximum)
        axis_element.attrib["default"] = int_or_float_to_str(axis.default)
        if axis.hidden:
            axis_element.attrib["hidden"] = "1"

        for language_code, label_name in sorted(axis.label_names.items()):
            label_element = ElementTree.Element("labelname")
            label_element.attrib[XML_LANG] = language_code
            label_element.text = label_name
            axis_element.append(label_element)

        for input_value, output_value in sorted(axis.mapping.items()):
            mapElement = ElementTree.Element("map")
            mapElement.attrib["input"] = int_or_float_to_str(input_value)
            mapElement.attrib["output"] = int_or_float_to_str(output_value)
            axis_element.append(mapElement)

        axes_element.append(axis_element)

    root.append(axes_element)


def _write_rules(
    rules: list[Rule], rules_processing_last: bool, root: ElementTree.Element
) -> None:
    if not rules:
        return

    rules_element = ElementTree.Element("rules")
    if rules_processing_last:
        rules_element.attrib["processing"] = "last"

    for rule in rules:
        if not rule.condition_sets:
            raise Error(f"Rule '{rule.name}' must have at least one condition set.")
        if not all(s.conditions for s in rule.condition_sets):
            raise Error(
                f"Rule '{rule.name}': all condition sets must have at least one condition."
            )
        if any(
            c.minimum is None and c.maximum is None
            for s in rule.condition_sets
            for c in s.conditions
        ):
            raise Error(
                f"Rule '{rule.name}': conditions must have either minimum, maximum or both set."
            )

        rule_element = ElementTree.Element("rule")
        rule_element.attrib["name"] = rule.name
        for condition_set in rule.condition_sets:
            conditionset_element = ElementTree.Element("conditionset")
            for condition in condition_set.conditions:
                condition_element = ElementTree.Element("condition")
                condition_element.attrib["name"] = condition.name
                if condition.minimum != -math.inf:
                    condition_element.attrib["minimum"] = int_or_float_to_str(
                        condition.minimum
                    )
                if condition.maximum != math.inf:
                    condition_element.attrib["maximum"] = int_or_float_to_str(
                        condition.maximum
                    )
                conditionset_element.append(condition_element)
            rule_element.append(conditionset_element)

        for sub_name, sub_with in sorted(rule.substitutions.items()):
            sub_element = ElementTree.Element("sub")
            sub_element.attrib["name"] = sub_name
            sub_element.attrib["with"] = sub_with
            rule_element.append(sub_element)

        rules_element.append(rule_element)

    root.append(rules_element)


def _write_sources(
    sources: list[Source],
    default_location: Location,
    root: ElementTree.Element,
) -> None:
    if not sources:
        return

    sources_element = ElementTree.Element("sources")

    for source in sources:
        source_element = ElementTree.Element("source")

        if source.filename is not None:
            source_element.attrib["filename"] = source.filename.as_posix()
        if source.name is not None:
            if not source.name.startswith("temp_master"):
                # do not save temporary source names
                source_element.attrib["name"] = source.name
        if source.family_name is not None:
            source_element.attrib["familyname"] = source.family_name
        if source.style_name is not None:
            source_element.attrib["stylename"] = source.style_name
        if source.layer_name is not None:
            source_element.attrib["layer"] = source.layer_name
        _write_location(source.location, default_location, source_element)

        sources_element.append(source_element)

    root.append(sources_element)


def _write_location(
    location: Location, default_location: Location, root: ElementTree.Element
) -> None:
    # Use the default location as a template and fill in the instance dimension values
    # whose axis names we know. Silently drop ones we don't know.
    location = {
        **default_location,
        **{k: v for k, v in location.items() if k in default_location},
    }

    if not location:
        return

    location_element = ElementTree.Element("location")

    for name, value in location.items():
        dimension_element = ElementTree.Element("dimension")
        dimension_element.attrib["name"] = name
        if isinstance(value, tuple):
            dimension_element.attrib["xvalue"] = int_or_float_to_str(value[0])
            dimension_element.attrib["yvalue"] = int_or_float_to_str(value[1])
        else:
            dimension_element.attrib["xvalue"] = int_or_float_to_str(value)
        location_element.append(dimension_element)

    root.append(location_element)


def _write_instances(
    instances: list[Instance], default_location: Location, root: ElementTree.Element
) -> None:
    if not instances:
        return

    instances_element = ElementTree.Element("instances")

    for instance in instances:
        instance_element = ElementTree.Element("instance")
        if instance.name is not None:
            if not instance.name.startswith("temp_instance"):
                instance_element.attrib["name"] = instance.name

        family_name = instance.family_name or instance.localised_family_name.get("en")
        if family_name is not None:
            instance_element.attrib["familyname"] = family_name

        style_name = instance.style_name or instance.localised_style_name.get("en")
        if style_name is not None:
            instance_element.attrib["stylename"] = style_name

        if instance.filename is not None:
            instance_element.attrib["filename"] = instance.filename.as_posix()

        smfn = (
            instance.style_map_family_name
            or instance.localised_style_map_family_name.get("en")
        )
        if smfn is not None:
            instance_element.attrib["stylemapfamilyname"] = smfn
        smsn = (
            instance.style_map_style_name
            or instance.localised_style_map_style_name.get("en")
        )
        if smsn is not None:
            instance_element.attrib["stylemapstylename"] = smsn

        for language_code, text in sorted(instance.localised_style_name.items()):
            if language_code == "en":
                continue  # Already stored in the element stylename attribute.
            element = ElementTree.Element("stylename")
            element.attrib[XML_LANG] = language_code
            element.text = text
            instance_element.append(element)

        for language_code, text in sorted(instance.localised_family_name.items()):
            if language_code == "en":
                continue  # Already stored in the element familyname attribute.
            element = ElementTree.Element("familyname")
            element.attrib[XML_LANG] = language_code
            element.text = text
            instance_element.append(element)

        for language_code, text in sorted(
            instance.localised_style_map_family_name.items()
        ):
            if language_code == "en":
                continue  # Already stored in the element stylename attribute.
            element = ElementTree.Element("stylemapfamilyname")
            element.attrib[XML_LANG] = language_code
            element.text = text
            instance_element.append(element)

        for language_code, text in sorted(
            instance.localised_style_map_style_name.items()
        ):
            if language_code == "en":
                continue  # Already stored in the element familyname attribute.
            element = ElementTree.Element("stylemapstylename")
            element.attrib[XML_LANG] = language_code
            element.text = text
            instance_element.append(element)

        if instance.postscript_font_name is not None:
            instance_element.attrib[
                "postscriptfontname"
            ] = instance.postscript_font_name

        _write_location(instance.location, default_location, instance_element)

        if instance.lib:
            _write_lib(instance.lib, instance_element)

        instances_element.append(instance_element)

    root.append(instances_element)


def _write_lib(lib: dict[str, Any], root: ElementTree.Element) -> None:
    if not lib:
        return

    lib_element = ElementTree.Element("lib")
    lib_element.append(fontTools.misc.plistlib.totree(lib, indent_level=4))
    root.append(lib_element)
