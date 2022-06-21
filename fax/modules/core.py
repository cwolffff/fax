import jax


class Module:

  # TODO: think about whether empty nodes is ok or could cause problems
  nodes = []

  def __init_subclass__(cls):
    jax.tree_util.register_pytree_node_class(cls)

  def tree_flatten(self):
    # assert len(self.nodes) != 0, "nodes must not be empty"
    children = tuple(
      getattr(self, param)
      for param in self.nodes
    )
    other_attrs = {
        name: value
        for name, value in vars(self).items()
        if name not 
        in self.nodes
    }
    return children, other_attrs

  @classmethod
  def tree_unflatten(cls, other_attrs, children):
    instance = cls.__new__(cls)
    for name, value in other_attrs.items():
      setattr(instance, name, value)
    for name, value in zip(instance.nodes, children):
      setattr(instance, name, value)
    return instance
  
  def __repr__(self):
    cls_name = type(self).__name__
    attrs = [
        f"{attr}={value}"
        for attr, value in vars(self).items()
        if attr not in self.nodes
    ]
    attrs_repr = ", ".join(attrs)
    return f"{cls_name}({attrs_repr})"
