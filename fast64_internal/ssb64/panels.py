from ..panels import SSB64_Panel
from ..utility import prop_split
from .operators import SSB64_ExportSkeleton, SSB64_ImportModel, SSB64_ImportSkeleton
from .properties import SSB64ExportSettings, SSB64ImportSettings


class SSB64_ImportModelPanel(SSB64_Panel):
    bl_idname = "SSB64_PT_import_model"
    bl_label = "SSB64 Import Model"
    bl_options = set()
    bl_order = 0

    def draw(self, context):
        col = self.layout.column()
        col.scale_y = 1.1

        col.operator(SSB64_ImportModel.bl_idname)
        col.operator(SSB64_ImportSkeleton.bl_idname)
        import_settings: SSB64ImportSettings = context.scene.fast64.ssb64.import_settings
        import_settings.draw_props(col)
        prop_split(col, context.scene.fast64.ssb64, "scale", "Scale")

        col.separator()
        col.operator(SSB64_ExportSkeleton.bl_idname)
        export_settings: SSB64ExportSettings = context.scene.fast64.ssb64.export_settings
        export_settings.draw_props(col)

        export_box = col.box().column()
        export_box.label(text="Exports XML resources for the active armature.")
