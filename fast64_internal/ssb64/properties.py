from pathlib import Path

import bpy
from bpy.props import BoolProperty, FloatProperty, PointerProperty, StringProperty
from bpy.types import Context, PropertyGroup, UILayout
from bpy.utils import register_class, unregister_class

from ..render_settings import on_update_render_settings
from ..utility import directory_ui_warnings, prop_split


class SSB64ImportSettings(PropertyGroup):
    decomp_path: StringProperty(
        name="Decomp Path",
        subtype="FILE_PATH",
        default="",
        description="Path to the Smash 64 decomp repo root",
    )
    model_name: StringProperty(
        name="Model",
        default="MarioModel",
        description="relocData model name, such as MarioModel or FoxModel",
    )
    dl_name: StringProperty(
        name="Display List",
        default="Joint_0x1668",
        description="Display list symbol/file stem to import from the relocData model",
    )
    remove_doubles: BoolProperty(name="Remove Doubles", default=True)
    import_normals: BoolProperty(name="Import Normals", default=True)

    @property
    def reloc_data_path(self) -> Path:
        return Path(bpy.path.abspath(self.decomp_path)) / "build" / "us" / "src" / "relocData"

    def draw_props(self, layout: UILayout):
        prop_split(layout, self, "decomp_path", "Decomp Path")
        directory_ui_warnings(layout, self.reloc_data_path)
        prop_split(layout, self, "model_name", "Model")
        prop_split(layout, self, "dl_name", "Display List")
        layout.prop(self, "remove_doubles")
        layout.prop(self, "import_normals")


class SSB64ExportSettings(PropertyGroup):
    output_dir: StringProperty(
        name="Export Directory",
        subtype="DIR_PATH",
        default="",
        description="Directory where the SSB64 extensionless XML resource family will be written",
    )
    model_name: StringProperty(
        name="Model",
        default="",
        description="Model name used in generated symbols, such as LinkModel or MarioModel",
    )
    internal_path: StringProperty(
        name="Internal Path",
        default="",
        description="Optional internal asset path prefix used in exported XML references",
    )
    scale: FloatProperty(
        name="Export Scale",
        default=100.0,
        min=0.0001,
        description="Scale used to convert Blender-space skeletons and vertices into Smash 64 game-space units",
    )

    @property
    def export_path(self) -> Path:
        path = bpy.path.abspath(self.output_dir)
        return Path(path) if path else Path()

    def draw_props(self, layout: UILayout):
        prop_split(layout, self, "output_dir", "Export Directory")
        if self.output_dir:
            directory_ui_warnings(layout, self.export_path)
        prop_split(layout, self, "model_name", "Model")
        prop_split(layout, self, "internal_path", "Internal Path")
        prop_split(layout, self, "scale", "Scale")


class SSB64_Properties(PropertyGroup):
    import_settings: PointerProperty(type=SSB64ImportSettings)
    export_settings: PointerProperty(type=SSB64ExportSettings)
    scale: FloatProperty(name="F3D Blender Scale", default=100, update=on_update_render_settings)

    @staticmethod
    def upgrade_changed_props():
        pass


ssb64_classes = (
    SSB64ImportSettings,
    SSB64ExportSettings,
    SSB64_Properties,
)


def ssb64_props_register():
    for cls in ssb64_classes:
        register_class(cls)


def ssb64_props_unregister():
    for cls in reversed(ssb64_classes):
        unregister_class(cls)
