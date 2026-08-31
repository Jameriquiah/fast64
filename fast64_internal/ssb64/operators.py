from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import bpy
import mathutils
from bpy.types import Operator

from ..f3d.f3d_gbi import get_F3D_GBI
from ..f3d.f3d_material import createF3DMat
from ..f3d.f3d_parser import get_include_data, importMeshC, parseMacroList
from ..utility import PluginError, applyRotation, hexOrDecInt, raisePluginError, readFile
from .exporter import (
    SSB64_BONE_DL_EXPR_PROP,
    SSB64_BONE_INDEX_PROP,
    SSB64_BONE_JOINT_TREE_PROP,
    SSB64_BONE_MODEL_PROP,
    export_joint_tree,
)
from .properties import SSB64ExportSettings, SSB64ImportSettings


ROOT_FILE_PATTERN = re.compile(r"^\d+_(.+)\.c$")
EXTERN_PREFIX_PATTERN = re.compile(r"^\s*extern\s+[^\n;]*\bd([A-Za-z0-9]+?)_", re.MULTILINE)
DOBJ_ARRAY_PATTERN = re.compile(r"DObjDesc\s+([A-Za-z0-9_]+)\s*\[\]\s*=\s*\{(.*?)\};", re.DOTALL)
DOBJ_ENTRY_PATTERN = re.compile(
    r"\{\s*"
    r"(\d+)\s*,\s*"
    r"\(void\*\)\s*([^,]+?)\s*,\s*"
    r"\{\s*([^\}]*)\}\s*,\s*"
    r"\{\s*([^\}]*)\}\s*,\s*"
    r"\{\s*([^\}]*)\}\s*"
    r"\}",
    re.DOTALL,
)
DISPLAY_LIST_PATTERN_TEMPLATE = r"Gfx\s*{name}\s*\[\s*\w*\s*\]\s*=\s*\{{([^\}}]*)\}}"
MOBJ_LIST_PATTERN = re.compile(r"MObjSub\s*\*\s+([A-Za-z0-9_]+)\s*\[\s*\d+\s*\]\s*=\s*\{(.*?)\};", re.DOTALL)
MOBJ_DISPATCH_PATTERN = re.compile(r"MObjSub\s*\*\*\s+([A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\};", re.DOTALL)
MOBJ_STRUCT_PATTERN = re.compile(r"MObjSub\s+([A-Za-z0-9_]+)\s*\[\s*\d+\s*\]\s*=\s*\{\s*\{(.*?)\}\s*\};", re.DOTALL)
DOBJ_DLLINK_PATTERN = re.compile(r"DObjDLLink\s+([A-Za-z0-9_]+)\s*\[\s*\d+\s*\]\s*=\s*\{(.*?)\};", re.DOTALL)

MOBJ_FLAG_NONE = 0
MOBJ_FLAG_ALPHA = 1 << 0
MOBJ_FLAG_SPLIT = 1 << 1
MOBJ_FLAG_PALETTE = 1 << 2
MOBJ_FLAG_FRAC = 1 << 4
MOBJ_FLAG_TEXTURE = 1 << 7
MOBJ_FLAG_PRIMCOLOR = 1 << 9
MOBJ_FLAG_ENVCOLOR = 1 << 10
MOBJ_FLAG_BLENDCOLOR = 1 << 11
MOBJ_FLAG_LIGHT1 = 1 << 12
MOBJ_FLAG_LIGHT2 = 1 << 13


@dataclass
class SSB64MObjSub:
    symbol: str
    fmt: str
    siz: str
    sprites: list[str]
    palettes: list[str]
    flags: int
    block_fmt: str
    block_siz: str
    block_dxt: int
    unk0A: int
    unk0C: int
    unk0E: int
    unk10: int
    unk38: int
    unk3A: int
    trau: float
    trav: float
    scau: float
    scav: float
    scrollu: float
    scrollv: float
    primcolor: tuple[int, int, int, int]
    prim_l: int
    prim_m: int
    envcolor: tuple[int, int, int, int]
    blendcolor: tuple[int, int, int, int]
    light1color: tuple[int, int, int, int]
    light2color: tuple[int, int, int, int]


def split_top_level_fields(data: str) -> list[str]:
    fields: list[str] = []
    depth_paren = 0
    depth_brace = 0
    start = 0

    for index, char in enumerate(data):
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace -= 1
        elif char == "," and depth_paren == 0 and depth_brace == 0:
            field = data[start:index].strip()
            if field:
                fields.append(field)
            start = index + 1

    tail = data[start:].strip()
    if tail:
        fields.append(tail)
    return fields


def strip_comments(value: str) -> str:
    return re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL).strip()


def parse_int_value(value: str) -> int:
    return int(strip_comments(value).removesuffix("f"), 0)


def parse_float_value(value: str) -> float:
    return float(strip_comments(value).removesuffix("f"))


def parse_color_value(value: str) -> tuple[int, int, int, int]:
    components = [component.strip() for component in strip_comments(value).strip("{} ").split(",")]
    return tuple(int(component, 0) for component in components[:4])


def parse_symbol_and_offset(expr: str) -> tuple[str | None, int]:
    param = strip_comments(expr)

    while param.startswith("(") and param.endswith(")"):
        inner = param[1:-1].strip()
        if not inner:
            break
        depth = 0
        balanced = True
        for char in inner:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        param = inner

    cast_match = re.match(r"^\(\s*[A-Za-z_][A-Za-z0-9_\s\*]*\)\s*(.+)$", param)
    while cast_match is not None:
        param = cast_match.group(1).strip()
        cast_match = re.match(r"^\(\s*[A-Za-z_][A-Za-z0-9_\s\*]*\)\s*(.+)$", param)

    if param == "NULL":
        return None, 0

    match = re.match(r"^\&?([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+\s*(.*))?$", param)
    if match is None:
        return None, 0

    symbol = match.group(1)
    offset = parse_int_value(match.group(2)) if match.group(2) is not None else 0
    return symbol, offset


def parse_pointer_symbol_list(data: str) -> dict[str, list[str]]:
    lists: dict[str, list[str]] = {}
    for match in MOBJ_LIST_PATTERN.finditer(data):
        fields = split_top_level_fields(match.group(2))
        entries: list[str] = []
        for field in fields:
            symbol, _ = parse_symbol_and_offset(field)
            entries.append(symbol if symbol is not None else "NULL")
        lists[match.group(1)] = entries
    return lists


def parse_mobj_sub_definitions(data: str) -> dict[str, SSB64MObjSub]:
    definitions: dict[str, SSB64MObjSub] = {}
    pointer_lists = parse_pointer_symbol_list(data)

    for match in MOBJ_STRUCT_PATTERN.finditer(data):
        fields = split_top_level_fields(match.group(2))
        if len(fields) < 40:
            continue

        sprite_list_symbol, _ = parse_symbol_and_offset(fields[3])
        palette_list_symbol, _ = parse_symbol_and_offset(fields[15])

        definitions[match.group(1)] = SSB64MObjSub(
            symbol=match.group(1),
            fmt=strip_comments(fields[1]),
            siz=strip_comments(fields[2]),
            sprites=[]
            if sprite_list_symbol is None or sprite_list_symbol == "NULL"
            else [entry for entry in pointer_lists.get(sprite_list_symbol, []) if entry != "NULL"],
            palettes=[]
            if palette_list_symbol is None or palette_list_symbol == "NULL"
            else [entry for entry in pointer_lists.get(palette_list_symbol, []) if entry != "NULL"],
            flags=parse_int_value(fields[16]),
            block_fmt=strip_comments(fields[17]),
            block_siz=strip_comments(fields[18]),
            block_dxt=parse_int_value(fields[19]),
            unk0A=parse_int_value(fields[5]),
            unk0C=parse_int_value(fields[6]),
            unk0E=parse_int_value(fields[7]),
            unk10=parse_int_value(fields[8]),
            unk38=parse_int_value(fields[21]),
            unk3A=parse_int_value(fields[22]),
            trau=parse_float_value(fields[9]),
            trav=parse_float_value(fields[10]),
            scau=parse_float_value(fields[11]),
            scav=parse_float_value(fields[12]),
            scrollu=parse_float_value(fields[23]),
            scrollv=parse_float_value(fields[24]),
            primcolor=parse_color_value(fields[28]),
            prim_l=parse_int_value(fields[29]),
            prim_m=parse_int_value(fields[30]),
            envcolor=parse_color_value(fields[32]),
            blendcolor=parse_color_value(fields[33]),
            light1color=parse_color_value(fields[34]),
            light2color=parse_color_value(fields[35]),
        )

    return definitions


def parse_mobj_dispatch_table(
    data: str,
    model_name: str,
    target_length: int | None = None,
) -> list[list[SSB64MObjSub] | None]:
    pointer_lists = parse_pointer_symbol_list(data)
    mobj_definitions = parse_mobj_sub_definitions(data)

    candidates: list[tuple[int, str, list[list[SSB64MObjSub] | None]]] = []
    preferred_name = f"d{model_name}_gap_0x0000"

    for match in MOBJ_DISPATCH_PATTERN.finditer(data):
        array_name = match.group(1)
        array_size = int(match.group(2), 0)
        fields = split_top_level_fields(match.group(3))
        dispatch: list[list[SSB64MObjSub] | None] = []

        for field in fields:
            symbol, offset = parse_symbol_and_offset(field)
            if symbol is None or symbol == "NULL":
                dispatch.append(None)
                continue

            if symbol in pointer_lists:
                start_index = offset // 4
                chain = [
                    mobj_definitions[entry]
                    for entry in pointer_lists[symbol][start_index:]
                    if entry != "NULL" and entry in mobj_definitions
                ]
                dispatch.append(chain or None)
            elif symbol in mobj_definitions:
                dispatch.append([mobj_definitions[symbol]])
            else:
                dispatch.append(None)

        if target_length is not None and array_size < target_length:
            continue

        score = sum(1 for chain in dispatch if chain is not None)
        if array_name == preferred_name:
            score += 10000
        candidates.append((score, array_name, dispatch))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][2]


def parse_dobj_dllink_targets(data: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for match in DOBJ_DLLINK_PATTERN.finditer(data):
        for entry in re.finditer(r"\{\s*\d+\s*,\s*\(Gfx\s*\*\)\s*([A-Za-z0-9_]+)\s*\}", match.group(2)):
            mapping[match.group(1)] = entry.group(1)
            break
    return mapping


def resolve_mobj_chain_index_for_dl(data: str, dl_symbol: str, dispatch_length: int) -> int | None:
    dllink_targets = parse_dobj_dllink_targets(data)
    for match in DOBJ_ARRAY_PATTERN.finditer(data):
        for index, entry in enumerate(DOBJ_ENTRY_PATTERN.finditer(match.group(2))):
            if index >= dispatch_length:
                return None

            dl_expr = entry.group(2).strip()
            dl_name, _ = decode_dl_expr(dl_expr)
            if dl_name == dl_symbol:
                return index
            if dl_name in dllink_targets and dllink_targets[dl_name] == dl_symbol:
                return index
    return None


class SSB64JointEntry:
    def __init__(
        self,
        depth: int,
        dl_expr: str,
        translation: tuple[float, float, float],
        rotation: tuple[float, float, float],
        scale: tuple[float, float, float],
        index: int,
    ):
        self.depth = depth
        self.dl_expr = dl_expr
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.index = index
        self.parent_index: int | None = None
        self.bone_name: str = f"joint_{index:02}"
        self.matrix = mathutils.Matrix.Identity(4)


def parse_float_triplet(values: str) -> tuple[float, float, float]:
    parts = [value.strip().removesuffix("f") for value in values.split(",")]
    return tuple(float(value) for value in parts[:3])


def get_reloc_data_root(import_settings: SSB64ImportSettings) -> Path:
    reloc_data_root = import_settings.reloc_data_path
    if not reloc_data_root.exists():
        raise PluginError(f"relocData folder not found: {reloc_data_root}")
    return reloc_data_root


def build_root_file_index(reloc_data_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in reloc_data_root.glob("*.c"):
        match = ROOT_FILE_PATTERN.match(path.name)
        if match is not None:
            index[match.group(1)] = path
    return index


def gather_reloc_data(reloc_name: str, root_file_index: dict[str, Path]) -> str:
    if reloc_name not in root_file_index:
        raise PluginError(f"Could not find relocData root file for '{reloc_name}'.")

    pending = [reloc_name]
    seen: set[str] = set()
    chunks: list[str] = []

    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)

        current_path = root_file_index.get(current)
        if current_path is None:
            raise PluginError(f"Missing relocData dependency '{current}'.")

        data = readFile(current_path)
        chunks.append(data)

        for dep_name in EXTERN_PREFIX_PATTERN.findall(data):
            if dep_name in root_file_index and dep_name not in seen:
                pending.append(dep_name)

    return "\n\n".join(chunks)


def resolve_display_list_symbol(data: str, model_name: str, dl_name: str) -> str:
    candidates = [
        dl_name,
        f"d{model_name}_{dl_name}",
        f"d{model_name}_{dl_name}_DisplayList",
    ]
    for candidate in candidates:
        if re.search(r"Gfx\s*" + re.escape(candidate) + r"\s*\[", data):
            return candidate
    raise PluginError(f"Could not find display list '{dl_name}' in relocData file '{model_name}'.")


def parse_joint_tree(data: str, model_name: str) -> list[SSB64JointEntry]:
    expected_name = f"d{model_name}_JointTree"
    matches = [match for match in DOBJ_ARRAY_PATTERN.finditer(data) if match.group(1) == expected_name]
    if not matches:
        raise PluginError(f"Could not find skeleton '{expected_name}'.")

    array_data = matches[-1].group(2)
    joints: list[SSB64JointEntry] = []
    for index, match in enumerate(DOBJ_ENTRY_PATTERN.finditer(array_data)):
        depth = int(match.group(1))
        dl_expr = match.group(2).strip()
        translation = parse_float_triplet(match.group(3))
        rotation = parse_float_triplet(match.group(4))
        scale = parse_float_triplet(match.group(5))
        joints.append(SSB64JointEntry(depth, dl_expr, translation, rotation, scale, index))

    if not joints:
        raise PluginError(f"Skeleton '{expected_name}' has no joints.")

    depth_stack: dict[int, int] = {}
    for joint in joints:
        joint.parent_index = depth_stack.get(joint.depth - 1)
        depth_stack[joint.depth] = joint.index
        for key in [key for key in depth_stack if key > joint.depth]:
            del depth_stack[key]

    return joints


def decode_dl_expr(dl_expr: str) -> tuple[str | None, int]:
    if dl_expr == "0x00000000":
        return None, 0

    direct_match = re.fullmatch(r"([A-Za-z0-9_]+)", dl_expr)
    if direct_match is not None:
        return direct_match.group(1), 0

    offset_match = re.fullmatch(r"\(\(u8 \*\)\s*([A-Za-z0-9_]+)\s*\+\s*(0x[0-9A-Fa-f]+|\d+)\)", dl_expr)
    if offset_match is not None:
        return offset_match.group(1), int(offset_match.group(2), 0)

    raise PluginError(f"Unhandled display list expression: {dl_expr}")


def parse_display_list_commands(dl_data: str, dl_name: str):
    pattern = re.compile(DISPLAY_LIST_PATTERN_TEMPLATE.format(name=re.escape(dl_name)))
    matches = list(pattern.finditer(dl_data))
    if not matches:
        raise PluginError(f"Cannot find display list named {dl_name}")

    dl_command_data = matches[-1].group(1)
    if "#include" in dl_command_data:
        dl_command_data = get_include_data(dl_command_data, strip=True)
    return parseMacroList(dl_command_data)


def create_armature_for_joints(name: str, joints: list[SSB64JointEntry]) -> bpy.types.Object:
    armature_data = bpy.data.armatures.new(f"{name}_armature")
    armature_data.display_type = "STICK"
    armature_obj = bpy.data.objects.new(f"{name}_armature", armature_data)
    armature_obj.show_in_front = True
    bpy.context.collection.objects.link(armature_obj)

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature_data.edit_bones

    for joint in joints:
        bone = edit_bones.new(joint.bone_name)
        head = joint.matrix.to_translation()
        tail = head + (joint.matrix.to_quaternion() @ mathutils.Vector((0.0, 0.3, 0.0)))
        bone.head = head
        bone.tail = tail
        if joint.parent_index is not None:
            bone.parent = edit_bones[joints[joint.parent_index].bone_name]

    bpy.ops.object.mode_set(mode="OBJECT")
    armature_obj[SSB64_BONE_MODEL_PROP] = name
    armature_obj[SSB64_BONE_JOINT_TREE_PROP] = f"d{name}_JointTree"
    for joint in joints:
        bone = armature_data.bones.get(joint.bone_name)
        if bone is None:
            continue
        bone[SSB64_BONE_INDEX_PROP] = joint.index
        bone[SSB64_BONE_DL_EXPR_PROP] = joint.dl_expr
    return armature_obj


def import_skeleton_model(
    model_name: str,
    dl_data: str,
    joints: list[SSB64JointEntry],
    armature_obj: bpy.types.Object,
    f3d_context,
    scale: float,
    remove_doubles: bool,
    import_normals: bool,
):
    for joint in joints:
        f3d_context.addMatrix(joint.bone_name, joint.matrix)
        f3d_context.limbToBoneName[joint.bone_name] = joint.bone_name
        if hasattr(f3d_context, "setCurrentMObjChainIndex"):
            f3d_context.setCurrentMObjChainIndex(joint.index)

        dl_name, dl_offset = decode_dl_expr(joint.dl_expr)
        if dl_name is None:
            continue

        commands = parse_display_list_commands(dl_data, dl_name)
        start_command = dl_offset // 8
        if start_command >= len(commands):
            continue

        f3d_context.setCurrentTransform(joint.bone_name)
        f3d_context.processCommands(dl_data, dl_name, commands, geometryStartIndex=start_command)

    mesh = bpy.data.meshes.new(model_name)
    mesh_obj = bpy.data.objects.new(model_name, mesh)
    bpy.context.collection.objects.link(mesh_obj)
    f3d_context.createMesh(mesh_obj, remove_doubles, import_normals, True)

    armature_modifier = mesh_obj.modifiers.new("Armature", "ARMATURE")
    armature_modifier.object = armature_obj
    mesh_obj.parent = armature_obj

    applyRotation([armature_obj, mesh_obj], -1.5707963267948966, "X")
    return armature_obj, mesh_obj


class SSB64_ImportModel(Operator):
    bl_idname = "scene.fast64_ssb64_import_model"
    bl_label = "Import Smash 64 Model"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        try:
            ssb64_props = context.scene.fast64.ssb64
            import_settings: SSB64ImportSettings = ssb64_props.import_settings

            reloc_data_root = get_reloc_data_root(import_settings)
            root_file_index = build_root_file_index(reloc_data_root)
            data = gather_reloc_data(import_settings.model_name, root_file_index)
            dl_symbol = resolve_display_list_symbol(data, import_settings.model_name, import_settings.dl_name)

            material = createF3DMat(None)
            material.f3d_mat.rdp_settings.set_rendermode = True

            mobj_dispatch = parse_mobj_dispatch_table(data, import_settings.model_name)
            material_chain_index = resolve_mobj_chain_index_for_dl(data, dl_symbol, len(mobj_dispatch))

            importMeshC(
                data,
                dl_symbol,
                ssb64_props.scale,
                import_settings.remove_doubles,
                import_settings.import_normals,
                "Opaque",
                create_ssb64_context(
                    reloc_data_root,
                    material,
                    data,
                    import_settings.model_name,
                    mobj_dispatch,
                    material_chain_index,
                ),
            )

            self.report({"INFO"}, "Success!")
            return {"FINISHED"}

        except Exception as exc:
            if context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            raisePluginError(self, exc)
            return {"CANCELLED"}


class SSB64_ImportSkeleton(Operator):
    bl_idname = "scene.fast64_ssb64_import_skeleton"
    bl_label = "Import Smash 64 Skeleton"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        try:
            ssb64_props = context.scene.fast64.ssb64
            import_settings: SSB64ImportSettings = ssb64_props.import_settings

            reloc_data_root = get_reloc_data_root(import_settings)
            root_file_index = build_root_file_index(reloc_data_root)
            data = gather_reloc_data(import_settings.model_name, root_file_index)
            joints = parse_joint_tree(data, import_settings.model_name)
            base_scale = mathutils.Matrix.Scale(1 / ssb64_props.scale, 4)

            for joint in joints:
                parent_matrix = joints[joint.parent_index].matrix if joint.parent_index is not None else base_scale
                local_matrix = (
                    mathutils.Matrix.Translation(mathutils.Vector(joint.translation))
                    @ mathutils.Euler(joint.rotation, "XYZ").to_matrix().to_4x4()
                )
                joint.matrix = parent_matrix @ local_matrix

                dl_name, _ = decode_dl_expr(joint.dl_expr)
                label = (
                    dl_name.removeprefix(f"d{import_settings.model_name}_")
                    if dl_name is not None
                    else f"node_{joint.index:02}"
                )
                joint.bone_name = f"{joint.index:02}_{label}"

            armature_obj = create_armature_for_joints(import_settings.model_name, joints)
            material = createF3DMat(None)
            material.f3d_mat.rdp_settings.set_rendermode = True
            mobj_dispatch = parse_mobj_dispatch_table(data, import_settings.model_name, len(joints))
            f3d_context = create_ssb64_context(
                reloc_data_root,
                material,
                data,
                import_settings.model_name,
                mobj_dispatch,
            )

            import_skeleton_model(
                import_settings.model_name,
                data,
                joints,
                armature_obj,
                f3d_context,
                ssb64_props.scale,
                import_settings.remove_doubles,
                import_settings.import_normals,
            )

            self.report({"INFO"}, "Success!")
            return {"FINISHED"}

        except Exception as exc:
            if context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            raisePluginError(self, exc)
            return {"CANCELLED"}


class SSB64_ExportSkeleton(Operator):
    bl_idname = "scene.fast64_ssb64_export_skeleton"
    bl_label = "Export Smash 64 Skeleton"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        try:
            armature_obj = context.active_object
            if armature_obj is None or armature_obj.type != "ARMATURE":
                raise PluginError("Armature not selected.")

            export_settings: SSB64ExportSettings = context.scene.fast64.ssb64.export_settings
            if not export_settings.output_dir.strip():
                raise PluginError("Set an export directory first.")
            export_dir = export_settings.export_path

            model_name = export_settings.model_name.strip()
            joint_tree_name, joint_tree_path, model_resource_name, model_resource_path = export_joint_tree(
                armature_obj,
                export_dir,
                model_name,
                export_settings.internal_path,
                export_settings.scale,
            )
            self.report(
                {"INFO"},
                f"Exported {joint_tree_name} and per-resource XML assets for {model_resource_name} to {joint_tree_path.parent.name}",
            )
            return {"FINISHED"}

        except Exception as exc:
            if context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            raisePluginError(self, exc)
            return {"CANCELLED"}


def create_ssb64_context(
    reloc_data_root: Path,
    material: bpy.types.Material,
    reloc_data_text: str,
    model_name: str,
    mobj_dispatch: list[list[SSB64MObjSub] | None] | None = None,
    initial_mobj_chain_index: int | None = None,
):
    from ..f3d.f3d_parser import F3DContext

    class SSB64F3DContext(F3DContext):
        def __init__(self, f3d, basePath, materialContext):
            super().__init__(f3d, basePath, materialContext)
            self.reloc_data_text = reloc_data_text
            self.model_name = model_name
            self.mobj_dispatch = mobj_dispatch or []
            self.active_mobj_chain = None
            if initial_mobj_chain_index is not None:
                self.setCurrentMObjChainIndex(initial_mobj_chain_index)

        def processTextureName(self, textureName: str) -> str:
            textureName = re.sub(r"/\*.*?\*/", "", textureName)
            return textureName.strip()

        def setCurrentMObjChainIndex(self, chain_index: int | None):
            if chain_index is None or chain_index < 0 or chain_index >= len(self.mobj_dispatch):
                self.active_mobj_chain = None
            else:
                self.active_mobj_chain = self.mobj_dispatch[chain_index]

        def _set_color_property(self, setter_name: str, color: tuple[int, int, int, int]):
            setattr(self.mat(), setter_name, True)
            linear = self.gammaInverseParam([str(component) for component in color])
            target_name = {
                "set_prim": "prim_color",
                "set_env": "env_color",
                "set_blend": "blend_color",
            }[setter_name]
            setattr(self.mat(), target_name, linear)

        def _apply_mobj_sub(self, mobj_sub: SSB64MObjSub):
            flags = mobj_sub.flags if mobj_sub.flags != MOBJ_FLAG_NONE else (MOBJ_FLAG_TEXTURE | 0x20 | MOBJ_FLAG_ALPHA)
            mat = self.mat()

            if flags & MOBJ_FLAG_PRIMCOLOR or flags & MOBJ_FLAG_FRAC or flags & 0x8:
                mat.prim_lod_min = mobj_sub.prim_m / 255.0
                mat.prim_lod_frac = mobj_sub.prim_l / 255.0
                self._set_color_property("set_prim", mobj_sub.primcolor)

            if flags & MOBJ_FLAG_ENVCOLOR:
                self._set_color_property("set_env", mobj_sub.envcolor)

            if flags & MOBJ_FLAG_BLENDCOLOR:
                self._set_color_property("set_blend", mobj_sub.blendcolor)

            if flags & MOBJ_FLAG_PALETTE and mobj_sub.palettes:
                palette_name = self.processTextureName(mobj_sub.palettes[0])
                self.tileSettings[5].fmt = "G_IM_FMT_RGBA"
                self.tileSettings[5].siz = "G_IM_SIZ_16b"
                self.tileSettings[5].line = 0
                self.tileSettings[5].tmem = 256
                self.currentTextureName = palette_name
                self.loadTexture(self.reloc_data_text, palette_name, [0, 0, 16, 16], self.tileSettings[5], True)
                self.tmemDict[256] = palette_name
                self.setTLUTMode("G_TT_RGBA16")
            else:
                self.setTLUTMode("G_TT_NONE")

            if flags & (MOBJ_FLAG_FRAC | MOBJ_FLAG_ALPHA | MOBJ_FLAG_TEXTURE) and mobj_sub.sprites:
                sprite_name = self.processTextureName(mobj_sub.sprites[0])
                tile = self.tileSettings[0]
                tile.fmt = mobj_sub.fmt
                tile.siz = mobj_sub.siz
                tile.line = 0
                tile.tmem = 0
                tile.palette = 0
                tile.cms = ("G_TX_NOMIRROR", "G_TX_WRAP")
                tile.cmt = ("G_TX_NOMIRROR", "G_TX_WRAP")
                tile.masks = 0
                tile.maskt = 0
                tile.shifts = 0
                tile.shiftt = 0
                self.currentTextureName = sprite_name
                self.loadTexture(self.reloc_data_text, sprite_name, None, tile, False)
                self.tmemDict[0] = sprite_name

            if flags & 0x20:
                tile = self.tileSizes[0]
                tile.uls = int((((mobj_sub.unk0C * mobj_sub.trau) + mobj_sub.unk0A) / max(mobj_sub.scau, 1e-6)) * 4.0)
                tile.ult = int(
                    (
                        ((((1.0 - mobj_sub.scav) - mobj_sub.trav) * mobj_sub.unk0E) + mobj_sub.unk0A)
                        / max(mobj_sub.scav, 1e-6)
                    )
                    * 4.0
                )
                tile.lrs = ((mobj_sub.unk0C - 1) << 2) + tile.uls
                tile.lrt = ((mobj_sub.unk0E - 1) << 2) + tile.ult

            if flags & 0x40:
                tile = self.tileSizes[1]
                tile.uls = int(
                    (((mobj_sub.unk38 * mobj_sub.scrollu) + mobj_sub.unk0A) / max(mobj_sub.scau, 1e-6)) * 4.0
                )
                tile.ult = int(
                    (
                        ((((1.0 - mobj_sub.scav) - mobj_sub.scrollv) * mobj_sub.unk3A) + mobj_sub.unk0A)
                        / max(mobj_sub.scav, 1e-6)
                    )
                    * 4.0
                )
                tile.lrs = ((mobj_sub.unk38 - 1) << 2) + tile.uls
                tile.lrt = ((mobj_sub.unk3A - 1) << 2) + tile.ult

            if flags & MOBJ_FLAG_TEXTURE:
                if mobj_sub.unk10 == 2:
                    s = (mobj_sub.unk0C * 64.0) / max(mobj_sub.scau, 1e-6)
                    t = (mobj_sub.unk0E * 64.0) / max(mobj_sub.scav, 1e-6)
                else:
                    s = (2097152.0 / max(mobj_sub.unk08, 1)) / max(mobj_sub.scau, 1e-6)
                    t = (2097152.0 / max(mobj_sub.unk08, 1)) / max(mobj_sub.scav, 1e-6)
                mat.tex_scale = [min(s, 0xFFFF) / 65535.0, min(t, 0xFFFF) / 65535.0]

            self.materialChanged = True

        def addVertices(self, num, start, vertexDataName, vertexDataOffset):
            super().addVertices(num, start, vertexDataName, vertexDataOffset)

            count = (
                int(num)
                if isinstance(num, int)
                else self.f3d.GBI_to_int(num)
                if hasattr(self.f3d, "GBI_to_int")
                else None
            )
            start_index = int(start) if isinstance(start, int) else None
            if count is None or start_index is None:
                from ..f3d.f3d_parser import math_eval

                count = math_eval(num, self.f3d)
                start_index = math_eval(start, self.f3d)

            if start_index <= 0:
                return

            if any(self.vertexBuffer[i] is not None for i in range(start_index)):
                return

            # Some Smash joint DLs assume slots 0..start-1 were populated by a prior
            # display list call. When importing a single DL standalone, mirror the first
            # loaded vertices into those empty slots so later gsSPModifyVertex / triangle
            # references have something to work with.
            mirror_count = min(start_index, count)
            for i in range(mirror_count):
                self.vertexBuffer[i] = self.vertexBuffer[start_index + i]

        def _resolve_include_dir(self, path: str) -> Path:
            include_path = Path(path.replace("\\", "/"))
            if len(include_path.parts) > 1:
                return reloc_data_root.joinpath(*include_path.parts[:-1])
            return reloc_data_root

        def getImagePathFromInclude(self, path, skip_base_path: bool = False):
            include_path = Path(path.replace("\\", "/"))
            include_dir = self._resolve_include_dir(path)
            prefix = include_path.name.split(".")[0]

            candidates = sorted(p for p in include_dir.glob(f"{prefix}*.png") if "jp_pad" not in p.name)
            if not candidates:
                raise PluginError(f"Could not find texture image for include '{path}'.")
            return str(candidates[0])

        def getVTXPathFromInclude(self, path):
            include_path = Path(path.replace("\\", "/"))
            return str(reloc_data_root.joinpath(*include_path.parts))

        def applyLights(self):
            # Smash imports should not create Blender light objects.
            return

        def processDLName(self, name: str):
            try:
                pointer = hexOrDecInt(name)
            except Exception:
                return name
            else:
                if pointer >> 24 == 0x0E and self.active_mobj_chain is not None:
                    branch_index = (pointer & 0x00FFFFFF) // 8
                    if 0 <= branch_index < len(self.active_mobj_chain):
                        self._apply_mobj_sub(self.active_mobj_chain[branch_index])
                    return None

                # Smash relocData often emits raw segmented DL calls that are not named C dlists.
                if pointer >> 24 != 0:
                    return None
                return name

    return SSB64F3DContext(get_F3D_GBI(), str(reloc_data_root), material)


ssb64_operator_classes = (SSB64_ImportModel, SSB64_ImportSkeleton, SSB64_ExportSkeleton)


def ssb64_operator_register():
    for cls in ssb64_operator_classes:
        bpy.utils.register_class(cls)


def ssb64_operator_unregister():
    for cls in reversed(ssb64_operator_classes):
        bpy.utils.unregister_class(cls)
