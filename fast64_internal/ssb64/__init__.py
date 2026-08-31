from bpy.utils import register_class, unregister_class

from .operators import ssb64_operator_register, ssb64_operator_unregister
from .panels import SSB64_ImportModelPanel
from .properties import SSB64_Properties, ssb64_props_register, ssb64_props_unregister


ssb64_panel_classes = (SSB64_ImportModelPanel,)


def ssb64_panel_register():
    for cls in ssb64_panel_classes:
        register_class(cls)


def ssb64_panel_unregister():
    for cls in reversed(ssb64_panel_classes):
        unregister_class(cls)


def ssb64_register(register_panels: bool):
    ssb64_props_register()
    if register_panels:
        ssb64_operator_register()
        ssb64_panel_register()


def ssb64_unregister(register_panels: bool):
    if register_panels:
        ssb64_panel_unregister()
        ssb64_operator_unregister()
    ssb64_props_unregister()
