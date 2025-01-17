import torch


class BasePass:
    @staticmethod
    def get_new_attr_name_with_prefix(
            module: torch.nn.Module,
            prefix: str,
            idx: int = 0):
        prefix = prefix.replace(".", "_")

        while True:
            attr_name = f"{prefix}{idx}"
            if not hasattr(module, attr_name):
                break
            idx += 1
        return attr_name

    @staticmethod
    def partition_module_name(target: str):
        *p, r = target.rsplit('.', 1)
        return '.'.join(p), r

    @staticmethod
    def get_module_of_node(gm, node, cls=tuple()):
        if node.op != 'call_module':
            return None
        mod = gm.get_submodule(node.target)
        if not cls or isinstance(mod, cls):
            return mod
        return None
