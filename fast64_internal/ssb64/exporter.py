from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import bpy
import mathutils

from ..f3d.f3d_gbi import DLFormat
from ..f3d import f3d_gbi
from ..hm64.f3d.soh_xml_exporter import _FModel_to_soh_xml
from ..hm64.f3d.f3d_texture_writer_hm64 import register as ensure_hm64_texture_writer
from ..hm64.f3d.hm64_f3d_writer import getInfoDict as hm64_getInfoDict
from ..hm64.f3d.soh_xml_exporter import register as ensure_hm64_soh_xml
from ..hm64.utility import sanitize_internal_asset_path, writeXMLData
from ..hm64.z64 import hm64_z64_f3d_writer
from ..hm64.z64.model_classes_hm64 import clear_hm64_material_state_cache
from ..utility import PluginError, toAlnum
from ..z64.model_classes import OOTModel
from ..f3d.f3d_gbi import GfxList, SPDisplayList, SPMatrix


SSB64_BONE_INDEX_PROP = "fast64_ssb64_joint_index"
SSB64_BONE_DL_EXPR_PROP = "fast64_ssb64_dl_expr"
SSB64_BONE_MODEL_PROP = "fast64_ssb64_model_name"
SSB64_BONE_JOINT_TREE_PROP = "fast64_ssb64_joint_tree_name"


@dataclass
class SSB64ExportJoint:
    bone: bpy.types.Bone
    order_index: int
    depth: int
    dl_expr: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float]
    scale: tuple[float, float, float]
    display_list_name: str | None


def _format_float(value: float) -> str:
    if abs(value) < 1e-9:
        value = 0.0
    return f"{value:.9g}"


def _get_default_model_name(armature_obj: bpy.types.Object) -> str:
    value = armature_obj.get(SSB64_BONE_MODEL_PROP)
    if isinstance(value, str) and value:
        return value
    return armature_obj.name.removesuffix("_armature")


def _get_joint_tree_name(armature_obj: bpy.types.Object, model_name: str) -> str:
    value = armature_obj.get(SSB64_BONE_JOINT_TREE_PROP)
    if isinstance(value, str) and value:
        return value
    return f"d{model_name}_JointTree"


def _get_object_path(model_name: str, internal_path: str) -> str:
    sanitized = sanitize_internal_asset_path(internal_path)
    return sanitized or model_name


def _get_export_target_dir(export_dir: Path, object_path: str) -> Path:
    target_dir = export_dir
    sanitized = sanitize_internal_asset_path(object_path)
    if sanitized:
        for part in sanitized.split("/"):
            target_dir = target_dir / part
    return target_dir


def _derive_dl_expr(bone: bpy.types.Bone, model_name: str) -> str:
    stored = bone.get(SSB64_BONE_DL_EXPR_PROP)
    if isinstance(stored, str):
        return stored

    label = bone.name
    if "_" in label and label[:2].isdigit():
        label = label[3:]

    if label.startswith("joint_") or label.startswith("node_"):
        return "0x00000000"

    if label.startswith(f"d{model_name}_"):
        return f"(void*){label}"

    if label.startswith("Joint_") or label.startswith("DL_"):
        if label.startswith("Joint_") and not label.endswith("_DisplayList"):
            label = f"{label}_DisplayList"
        return f"(void*)d{model_name}_{label}"

    return "0x00000000"


def _iter_depth_first_bones(armature_obj: bpy.types.Object) -> list[bpy.types.Bone]:
    def sort_key(bone: bpy.types.Bone):
        index = bone.get(SSB64_BONE_INDEX_PROP)
        return (0, int(index), bone.name) if isinstance(index, int) else (1, bone.name)

    ordered: list[bpy.types.Bone] = []

    def visit(bone: bpy.types.Bone):
        ordered.append(bone)
        for child in sorted(bone.children, key=sort_key):
            visit(child)

    roots = sorted((bone for bone in armature_obj.data.bones if bone.parent is None), key=sort_key)
    for root in roots:
        visit(root)
    return ordered


def _get_export_bones(armature_obj: bpy.types.Object) -> list[bpy.types.Bone]:
    bones = list(armature_obj.data.bones)
    if not bones:
        raise PluginError("Armature has no bones to export.")

    indexed_bones = [bone for bone in bones if isinstance(bone.get(SSB64_BONE_INDEX_PROP), int)]
    if len(indexed_bones) == len(bones):
        sorted_bones = sorted(bones, key=lambda bone: (int(bone[SSB64_BONE_INDEX_PROP]), bone.name))
        indices = [int(bone[SSB64_BONE_INDEX_PROP]) for bone in sorted_bones]
        if indices == list(range(len(sorted_bones))):
            return sorted_bones

    return _iter_depth_first_bones(armature_obj)


def _local_matrix_for_bone(bone: bpy.types.Bone) -> mathutils.Matrix:
    if bone.parent is None:
        return bone.matrix_local.copy()
    return bone.parent.matrix_local.inverted() @ bone.matrix_local


def _decode_dl_expr(dl_expr: str) -> tuple[str | None, int]:
    if dl_expr == "0x00000000":
        return None, 0

    direct = dl_expr.removeprefix("(void*)").strip()
    if direct and direct[0].isalpha():
        return direct, 0

    offset_prefix = "(void*)((u8 *)"
    if dl_expr.startswith(offset_prefix) and dl_expr.endswith(")"):
        inner = dl_expr[len(offset_prefix) : -1]
        if " + " in inner:
            symbol, offset = inner.split(" + ", 1)
            return symbol.strip(), int(offset.strip(), 0)

    return None, 0


def _build_export_joints(
    armature_obj: bpy.types.Object,
    model_name: str,
    display_list_map: dict[str, str | None],
    export_scale: float,
) -> list[SSB64ExportJoint]:
    ordered_bones = _get_export_bones(armature_obj)
    joints: list[SSB64ExportJoint] = []

    for order_index, bone in enumerate(ordered_bones):
        local_matrix = _local_matrix_for_bone(bone)
        translation, rotation, scale = local_matrix.decompose()
        depth = 0
        parent = bone.parent
        while parent is not None:
            depth += 1
            parent = parent.parent

        rotation_euler = rotation.to_euler("XYZ")
        translation_values = tuple(float(value) * export_scale for value in translation)
        joints.append(
            SSB64ExportJoint(
                bone=bone,
                order_index=order_index,
                depth=depth,
                dl_expr=_derive_dl_expr(bone, model_name),
                translation=translation_values,
                rotation=tuple(float(value) for value in rotation_euler),
                scale=tuple(float(value) for value in scale),
                display_list_name=display_list_map.get(bone.name),
            )
        )

    return joints


def _build_joint_tree_xml(joint_tree_name: str, object_path: str, joints: list[SSB64ExportJoint]) -> str:
    lines = [f'<SSB64JointTree Version="0" Name="{joint_tree_name}" JointCount="{len(joints)}">']
    for joint in joints:
        raw_dl_name, raw_dl_offset = _decode_dl_expr(joint.dl_expr)
        attrs = [
            f'Index="{joint.order_index}"',
            f'Bone="{joint.bone.name}"',
            f'Depth="{joint.depth}"',
            f'TransX="{_format_float(joint.translation[0])}"',
            f'TransY="{_format_float(joint.translation[1])}"',
            f'TransZ="{_format_float(joint.translation[2])}"',
            f'RotX="{_format_float(joint.rotation[0])}"',
            f'RotY="{_format_float(joint.rotation[1])}"',
            f'RotZ="{_format_float(joint.rotation[2])}"',
            f'ScaleX="{_format_float(joint.scale[0])}"',
            f'ScaleY="{_format_float(joint.scale[1])}"',
            f'ScaleZ="{_format_float(joint.scale[2])}"',
            f'RawDisplayListExpr="{joint.dl_expr}"',
            f'RawDisplayListOffset="{raw_dl_offset}"',
        ]
        if raw_dl_name is not None:
            attrs.append(f'RawDisplayListSymbol="{raw_dl_name}"')
        if joint.display_list_name is not None:
            attrs.append(f'DisplayList="{object_path}/{joint.display_list_name}"')
        lines.append("\t<Joint " + " ".join(attrs) + "/>")
    lines.append("</SSB64JointTree>")
    return "\n".join(lines)


@contextmanager
def _use_hm64_skeleton_material_writer():
    from ..z64.exporter.skeleton import functions as shared_skeleton_functions

    old_get_info_dict = shared_skeleton_functions.getInfoDict
    old_process_vertex_group = shared_skeleton_functions.ootProcessVertexGroup
    shared_skeleton_functions.getInfoDict = hm64_getInfoDict
    shared_skeleton_functions.ootProcessVertexGroup = hm64_z64_f3d_writer.ootProcessVertexGroup
    try:
        yield shared_skeleton_functions
    finally:
        shared_skeleton_functions.getInfoDict = old_get_info_dict
        shared_skeleton_functions.ootProcessVertexGroup = old_process_vertex_group


def _require_mesh_child(armature_obj: bpy.types.Object):
    mesh_children = [child for child in armature_obj.children if child.type == "MESH"]
    if not mesh_children:
        raise PluginError("SSB64 XML export needs the armature's mesh child so it can emit display lists.")
    return mesh_children[0]


def _collect_revert_lists(f_model: OOTModel) -> set[int]:
    revert_ids: set[int] = set()
    for f_material, _ in f_model.materials.values():
        if f_material.revert is not None:
            revert_ids.add(id(f_material.revert))
            f_material.revert = None
    return revert_ids


def _strip_ssb64_unsafe_commands(gfx_list: GfxList | None, revert_ids: set[int], visited: set[int]):
    if gfx_list is None or id(gfx_list) in visited:
        return
    visited.add(id(gfx_list))

    filtered_commands = []
    for command in gfx_list.commands:
        if isinstance(command, SPMatrix):
            continue
        if isinstance(command, SPDisplayList) and id(command.displayList) in revert_ids:
            continue
        filtered_commands.append(command)
    gfx_list.commands = filtered_commands

    for command in gfx_list.commands:
        nested_list = getattr(command, "displayList", None)
        if isinstance(nested_list, GfxList):
            _strip_ssb64_unsafe_commands(nested_list, revert_ids, visited)


def _sanitize_fmodel_for_ssb64_xml(f_model: OOTModel):
    revert_ids = _collect_revert_lists(f_model)
    visited: set[int] = set()

    for mesh in f_model.meshes.values():
        _strip_ssb64_unsafe_commands(mesh.draw, revert_ids, visited)
        for tri_group in mesh.triangleGroups:
            _strip_ssb64_unsafe_commands(tri_group.triList, revert_ids, visited)
        for draw_override in mesh.draw_overrides:
            _strip_ssb64_unsafe_commands(draw_override.draw, revert_ids, visited)

    for f_material, _ in f_model.materials.values():
        _strip_ssb64_unsafe_commands(f_material.material, revert_ids, visited)

    for texture in f_model.textures.values():
        texture_name = getattr(texture, "name", "")
        if isinstance(texture_name, str):
            for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
                if texture_name.lower().endswith(suffix):
                    texture.name = texture_name[: -len(suffix)]
                    break
        texture_filename = getattr(texture, "filename", "")
        if isinstance(texture_filename, str):
            for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
                if texture_filename.lower().endswith(suffix):
                    texture.filename = texture_filename[: -len(suffix)]
                    break


def export_joint_tree(
    armature_obj: bpy.types.Object,
    export_dir: Path,
    model_name: str,
    internal_path: str = "",
    export_scale: float = 100.0,
):
    if armature_obj.type != "ARMATURE":
        raise PluginError("Armature not selected.")

    _require_mesh_child(armature_obj)

    if not model_name:
        model_name = _get_default_model_name(armature_obj)

    export_dir.mkdir(parents=True, exist_ok=True)
    joint_tree_name = _get_joint_tree_name(armature_obj, model_name)
    object_path = _get_object_path(model_name, internal_path)
    target_dir = _get_export_target_dir(export_dir, object_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    ensure_hm64_soh_xml()
    ensure_hm64_texture_writer()

    f_model = OOTModel(model_name, DLFormat.Static, "Opaque")
    try:
        with _use_hm64_skeleton_material_writer() as shared_skeleton_functions:
            original_get_fmesh_name = f3d_gbi.getFMeshName
            f3d_gbi.getFMeshName = lambda vertex_group, name_prefix, draw_layer, is_skinned: (
                toAlnum(name_prefix + ("_" if name_prefix != "" else "") + vertex_group)
                + ("_skinned" if is_skinned else "")
            )
            try:
                skeleton, f_model = shared_skeleton_functions.ootConvertArmatureToSkeletonWithMesh(
                    armature_obj,
                    mathutils.Matrix.Scale(export_scale, 4),
                    f_model,
                    model_name,
                    True,
                    "Opaque",
                    False,
                )
            finally:
                f3d_gbi.getFMeshName = original_get_fmesh_name

        _sanitize_fmodel_for_ssb64_xml(f_model)

        _FModel_to_soh_xml(
            f_model,
            str(target_dir),
            object_path,
            include_cull_vertices=False,
            combine_root_meshes=False,
        )

        limb_list = skeleton.createLimbList()
        display_list_map = {limb.boneName: limb.DL.name if limb.DL is not None else None for limb in limb_list}
        joints = _build_export_joints(armature_obj, model_name, display_list_map, export_scale)
        joint_tree_xml = _build_joint_tree_xml(joint_tree_name, object_path, joints)
        writeXMLData(joint_tree_xml, str(target_dir / joint_tree_name))
    finally:
        clear_hm64_material_state_cache(f_model)

    return joint_tree_name, target_dir / joint_tree_name, model_name, None
