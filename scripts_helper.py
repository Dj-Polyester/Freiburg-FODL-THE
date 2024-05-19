from typing import Tuple, Sequence, Callable
import math
import itertools
from functools import reduce
from sympy import Symbol, Eq, simplify
from sympy.solvers import solveset, S
import torch
from torch import Tensor, tensor, special, nn
from torch.nn import functional as F

# Background


class BLR:
    """
    Class implementing Bayesian Linear Regression as described in the lecture slides
    __init__ for the class takes prior mean, prior variance, noise and bias (for DNGO)
    as input.

    _mu_pre is the prior mean, _sigma_pre is the prior covariance,
    _noise is the noise variance sigma_n^2 in the input data
    """

    def __init__(
        self,
        mu_pre: Tensor | Sequence = None,
        sigma_pre: Tensor | Sequence = None,
        size_mu_pre=None,
        size_sigma_pre=None,
        noise: float = 1,
        bias: bool = False,
    ):
        # self.mu_pre = _set_tensor_size(mu_pre, size_mu_pre)
        self.mu_pre = torch.zeros(size_mu_pre)
        self.sigma_pre = _set_tensor_size(sigma_pre, size_sigma_pre)
        # if torch.count_nonzero(self.mu_pre) > 0:
        #     raise NotImplementedError("BLR currently only supports mu_pre=0")
        self.noise = noise
        self.bias = _clone_from_tensor_or_other(bias)
        self.mu_post = None
        self.sigma_post = None

    def fit(
        self,
        X: Tensor | Sequence = None,
        y: Tensor | Sequence = None,
        size_X: Tensor | Sequence = None,
        size_y: Tensor | Sequence = None,
    ) -> Tuple[Tensor | Sequence, Tensor | Sequence]:
        X = _set_tensor_size(X, size_X)
        y = _set_tensor_size(y, size_y)
        if self.bias:
            bias = torch.ones((X.shape[0], 1))
            X = torch.hstack([X, bias])

        # Using Dxn format used in the GPML book
        X = X.T

        Sigma_pre_inv = torch.linalg.inv(self.sigma_pre)  # Prior
        A = torch.matmul(X, X.T) / self.noise + Sigma_pre_inv
        self.sigma_post = torch.linalg.inv(A)

        mu_post_ = torch.matmul(X, y) / self.noise
        mu_post = torch.matmul(self.sigma_post, mu_post_)
        self.mu_post = mu_post

        return self.mu_post, self.sigma_post

    def predict(
        self,
        X: Tensor | Sequence = None,
        size_X=None,
    ) -> Tuple[Tensor | Sequence, Tensor | Sequence]:
        X = _set_tensor_size(X, size_X)
        if self.bias:
            bias = torch.ones((X.shape[0], 1))
            X = torch.hstack([X, bias])

        X = X.T
        x_mean = torch.matmul(X.T, self.mu_post)
        x_std = torch.matmul(torch.matmul(X.T, self.sigma_post), X)
        x_std = torch.diagonal(x_std)
        x_mean = x_mean.reshape(-1, 1)
        x_std = x_std.reshape(-1, 1)

        return x_mean, x_std


def prod(seq: Sequence):
    return reduce(lambda x, y: x * y, seq)


def _solve(**kwargs):
    lcls = locals()
    for k, v in kwargs.items():
        lcls[k] = Symbol(k) if v == None else v

    lrhs = kwargs["eq"].split("=")
    if len(lrhs) == 1:
        eq = eval(lrhs)
    elif len(lrhs) == 2:
        eq = Eq(eval(lrhs[0]), eval(lrhs[1]))
    else:
        raise ValueError(f"The equation string either has right or left side")

    eq = simplify(eq)
    eqvars = tuple(lcls[k] for k, v in kwargs.items() if v == None)
    if len(eqvars) != 1:
        raise ValueError(f"Number of unknown elements should be one")
    if "domain" not in kwargs or kwargs["domain"] == None:
        raise ValueError(f"The parameter domain should be assigned a value")
    return solveset(eq, eqvars[0], domain=kwargs["domain"])


def _set_tensor_size(
    _tensor,
    size,
    dtype=torch.float32,
    max_val_long: int = None,
):
    if _tensor == None:
        if size == None:
            raise ValueError("size cant be None")
        return _auto_tensor(size, dtype, max_val_long)
    else:
        return _clone_from_tensor_or_other(_tensor, dtype=dtype)


def _print_dic(dic: dict, large_elems: dict = {}):
    dic.update(large_elems)
    for k, v in dic.items():
        print(f"{k}: {v}")


def _print_size_numel_tensor(_tensor: Tensor, name: str, print_tensors=False):
    _print_dic(
        {
            f"{name} size": f"{list(_tensor.shape)}, {_tensor.numel()}",
        },
        large_elems={name: f"\n{_tensor}"} if print_tensors else {},
    )


def _print_size_numel_grad(module, print_grad=False, print_tensors=False):
    param_attrs = ["weight", "bias"]
    total = 0
    for attr in param_attrs:
        if hasattr(module, attr) and getattr(module, attr) != None:
            _param = getattr(module, attr)
            _tensor = _param.data
            _print_size_numel_tensor(_tensor, attr, print_tensors)
            if print_grad:
                grad = _param.grad
                _print_size_numel_tensor(grad, f"{attr} grad", print_tensors)
            total += _tensor.numel()

    _print_dic({"total parameters": total})

    return total


def _print_ops_ret_out(out_cache_attr_name, module):
    output: Tensor = getattr(module, out_cache_attr_name)
    mult = None
    add = None
    module_name_lowered: str = type(module).__name__.lower()
    total_linear_ops = 0
    total_nonlinear_ops = 0
    if module_name_lowered == "linear":
        mult = output.numel() * module.weight.data.shape[1]
        add = 0 if module.bias == None else output.numel()
    elif module_name_lowered.startswith("conv"):
        mult = output.numel() * prod(module.weight.shape[1:])
        add = 0 if module.bias == None else output.numel()
    elif (
        "sigmoid" in module_name_lowered
        or "tanh" in module_name_lowered
        or "elu" in module_name_lowered
        or "swish" in module_name_lowered
    ):
        total_nonlinear_ops = output.numel()
    if mult != None and add != None:
        total_linear_ops = mult + add
        _print_dic(
            {
                "multiplications": mult,
                "additions": add,
                "total operations": total_linear_ops,
            },
        )
    else:
        print(f"total nonlinear operations: {total_nonlinear_ops}")
    return output, total_linear_ops, total_nonlinear_ops


def _clone_from_tensor_or_other(_input: Sequence | Tensor, dtype=torch.float32):
    return (
        _input.clone().type(dtype)
        if isinstance(_input, Tensor)
        else tensor(_input, dtype=dtype)
    )


class DebugNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.out_cache_attr_name = "__cached_out"
        self.in_cache_attr_name = "__cached_in"
        self.setup(*args, **kwargs)
        _register_forward_hooks_of_modules(self, self.out_cache_attr_name)

    def setup(self, *args, **kwargs):
        return NotImplementedError()

    def forward(self, x: Sequence | Tensor = None, size=None, print_tensors=False):
        x = _set_tensor_size(x, size)
        _print_size_numel_tensor(x, "input", print_tensors)
        x = _compensate_dim(x, 1)
        out = self._forward(x)
        return out, _size_numel_tensor(
            self,
            self.out_cache_attr_name,
            print_tensors,
            print_grad=False,
        )

    def backward(
        self,
        x: Sequence | Tensor = None,
        y: Sequence | Tensor = None,
        size_x=None,
        size_y=None,
        max_y: int = None,
        type_y=torch.long,
        print_tensors=False,
        lossfn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ):
        x = _set_tensor_size(x, size_x)
        if isinstance(size_y, (int, float)):
            size_y = (size_y,)
        y = _set_tensor_size(y, size_y, max_val_long=max_y, dtype=type_y)
        _print_size_numel_tensor(x, "input", print_tensors)
        x = _compensate_dim(x, 1)
        y = _compensate_dim(y, 1)
        out = self._forward(x)
        if lossfn != None:
            out = lossfn(out, y)
        out.backward()
        return out, _size_numel_tensor(
            self,
            self.out_cache_attr_name,
            print_tensors,
        )

    def _forward(self, x) -> Tensor:
        raise NotImplementedError()

    def size_numel(self, prefix=""):
        return _size_numel_tensor(
            self,
            self.out_cache_attr_name,
            print_tensors=False,
            calc_forward=False,
            print_grad=False,
            prefix=prefix,
        )


def _register_forward_hooks_of_modules(self: nn.Module, out_cache_attr_name: str):
    def setattr_ifnothave(module, attr, val):
        setattr(module, attr, val)

    def forward_hook(module, input, output):
        setattr_ifnothave(module, out_cache_attr_name, output)

    for module in self._modules.values():
        if hasattr(module, "_modules") and len(module._modules) != 0:
            _register_forward_hooks_of_modules(module, out_cache_attr_name)
        else:
            module.register_forward_hook(forward_hook)


def _size_numel_tensor(
    self: nn.Module,
    out_cache_attr_name: str,
    print_tensors=True,
    calc_forward=True,
    print_grad=True,
    prefix="",
):
    grand_total_params = 0
    grand_total_linear_ops = 0
    grand_total_nonlinear_ops = 0
    grand_total_ops = None
    for i, (name, module) in enumerate(self._modules.items()):
        if hasattr(module, "_modules") and len(module._modules) != 0:
            total_params, total_linear_ops, total_nonlinear_ops, _ = _size_numel_tensor(
                module,
                out_cache_attr_name,
                print_tensors,
                calc_forward=calc_forward,
                print_grad=print_grad,
                prefix=f"{prefix}{name}.",
            )
            grand_total_params += total_params
            grand_total_linear_ops += total_linear_ops
            grand_total_nonlinear_ops += total_nonlinear_ops
        else:
            module_name = type(module).__name__
            print(f"\nop name {i+1}: {prefix}{name} ({module_name})")
            total = _print_size_numel_grad(module, print_grad, print_tensors)
            grand_total_params += total
            if calc_forward:
                output, total_linear_ops, total_nonlinear_ops = _print_ops_ret_out(
                    out_cache_attr_name, module
                )
                _print_size_numel_tensor(output, "output", print_tensors)
                grand_total_linear_ops += total_linear_ops
                grand_total_nonlinear_ops += total_nonlinear_ops
    if prefix == "":
        print(f"\ngrand total number of params: {grand_total_params}")
        if calc_forward:
            grand_total_ops = grand_total_linear_ops + grand_total_nonlinear_ops
            print(f"grand total number of linear ops: {grand_total_linear_ops}")
            print(f"grand total number of nonlinear ops: {grand_total_nonlinear_ops}")
            print(f"grand total number of ops: {grand_total_ops}")
    return (
        grand_total_params,
        grand_total_linear_ops,
        grand_total_nonlinear_ops,
        grand_total_ops,
    )


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int = None,
        out_channels: int = None,
        kernel_size: int | Tuple[int] = None,
        stride: int | Tuple[int] = 1,
        padding: str | int | Tuple[int] = 0,
        dilation: int | Tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.conv_depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.conv1_1 = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(self, x):
        return self.conv1_1(self.conv_depthwise(x))


class SingleLayerNet(DebugNet):
    def setup(self, net: nn.Module, name="net"):
        self.name = name
        setattr(self, name, net)

    def _forward(self, x):
        return getattr(self, self.name)(x)


DEFAULTS = {
    "stride": 1,
    "padding": 0,
    "dilation": 1,
}


def binary_onehot(input):
    return torch.hstack((1 - input[:, None], input[:, None]))


def index_onehot(input, num_classes):
    return torch.eye(num_classes)[input]


def entropy_with_logits(
    y: Tensor,
    y_hat: Tensor = None,
    num_classes=None,
    reduce="mean",
    logits="softmax",
    _type="cross",
):
    if y_hat != None and num_classes == None:
        num_classes = y_hat.shape[-1]
    if len(y.shape) == 1:
        y = index_onehot(y, num_classes)
    elif len(y.shape) != 2:
        raise ValueError()
    if y_hat != None:
        y_hat = y_hat.type(torch.float64)
    y = y.type(torch.float64)
    if logits == "softmax":
        y_hat = F.softmax(y_hat, dim=1)
    elif logits == "sigmoid":
        y_hat = F.sigmoid(y_hat)
    elif logits != None:
        raise ValueError("Wrong logits")
    if num_classes == 2 and y_hat != None and len(y_hat.shape) == 1:
        y_hat = binary_onehot(y_hat)
    if y_hat != None and y_hat.shape != y.shape:
        raise ValueError("Lengths not equal")
    logterm = None
    if _type == "cross":
        logterm = -y_hat.log()
    if _type == "info":
        logterm = -y.log()
    if _type == "kl":
        logterm = y.log() - y_hat.log()
    nonreduced = y * logterm
    nonreduced_nan_mask = nonreduced != nonreduced
    nonreduced[nonreduced_nan_mask] = 0
    if reduce == "none":
        return nonreduced
    if y_hat != None and len(y_hat.shape) == 1:
        return nonreduced.sum()
    nonreduced_sum = nonreduced.sum(dim=1)
    return getattr(nonreduced_sum, reduce)(dim=0)


def e(y, num_classes=None, reduce="mean"):
    return entropy_with_logits(
        y, num_classes=num_classes, reduce=reduce, logits=None, _type="info"
    )


def bce(y_hat, y, reduce="mean"):
    return ce(y_hat, y, 2, reduce)


def ce(y_hat, y, num_classes=None, reduce="mean"):
    return entropy_with_logits(y, y_hat, num_classes, reduce, logits=None)


def kl(y_hat, y, num_classes=None, reduce="mean"):
    return entropy_with_logits(y, y_hat, num_classes, reduce, logits=None, _type="kl")


def bce_with_logits(y_hat, y, reduce="mean"):
    return entropy_with_logits(y, y_hat, 2, reduce, logits="sigmoid")


def ce_with_logits(y_hat, y, num_classes, reduce="mean"):
    return entropy_with_logits(y, y_hat, num_classes, reduce)


def ultimate_e(y, num_classes=None, reduce="mean"):
    """y is a 2D tensor"""
    y = _clone_from_tensor_or_other(y).float()
    nonreduced = special.entr(y)
    reduced = None
    if reduce == "none":
        reduced = nonreduced
    else:
        nonreduced = nonreduced.sum(dim=1)
        reduced = getattr(nonreduced, reduce)(dim=0)
    return e(y, num_classes, reduce), reduced


def ultimate_bce(y_hat, y, reduce="mean"):
    """
    Both tensors are 1D
    y_hat: predictions of y being 1
    """
    y_hat, y = (
        _clone_from_tensor_or_other(y_hat).float(),
        _clone_from_tensor_or_other(y).float(),
    )
    y_ = y.argmax(dim=1) if len(y.shape) > 1 else y
    return (
        bce(y_hat, y, reduce),
        F.binary_cross_entropy(
            input=y_hat.type(torch.float64),
            target=y.type(torch.float64),
            reduction=reduce,
        ),
        F.nll_loss(input=torch.log(binary_onehot(y_hat)), target=y_, reduction=reduce),
    )


def ultimate_ce(y_hat, y, num_classes=None, reduce="mean"):
    """y is a 1D tensor"""
    y_hat, y = (
        _clone_from_tensor_or_other(y_hat).float(),
        _clone_from_tensor_or_other(y).float(),
    )
    y_ = y.argmax(dim=1) if len(y.shape) > 1 else y
    return (
        ce(y_hat, y, num_classes, reduce),
        F.nll_loss(input=torch.log(y_hat), target=y_, reduction=reduce),
    )


def ultimate_kl(y_hat, y, num_classes=None, reduce="mean"):
    """y is a 1D tensor"""
    y_hat, y = (
        _clone_from_tensor_or_other(y_hat).float(),
        _clone_from_tensor_or_other(y).float(),
    )
    y_ = y if len(y.shape) > 1 else index_onehot(y, num_classes=y_hat.shape[-1])
    return (
        kl(y_hat, y, num_classes, reduce),
        ce(y_hat, y, num_classes, reduce)
        - e(y, num_classes=y_hat.shape[-1], reduce=reduce),
        F.kl_div(
            input=y_hat.type(torch.float64).log(),
            target=y_.log(),
            reduction="batchmean" if reduce == "mean" else reduce,
            log_target=True,
        ),
    )


def ultimate_bce_with_logits(y_hat, y, reduce="mean"):
    """
    Both tensors are 1D
    y_hat: probability predictions of y being 1
    """
    y_hat, y = (
        _clone_from_tensor_or_other(y_hat).float(),
        _clone_from_tensor_or_other(y).float(),
    )
    y_ = y.argmax(dim=1) if len(y.shape) > 1 else y
    return (
        bce_with_logits(y_hat, y, reduce),
        F.binary_cross_entropy_with_logits(
            input=y_hat.type(torch.float64),
            target=y.type(torch.float64),
            reduction=reduce,
        ),
        F.nll_loss(
            input=torch.log(binary_onehot(F.sigmoid(y_hat.type(torch.float64)))),
            target=y_,
            reduction=reduce,
        ),
    )


def ultimate_ce_with_logits(y_hat, y, num_classes=None, reduce="mean"):
    """y is a 1D tensor"""
    y_hat, y = (
        _clone_from_tensor_or_other(y_hat).float(),
        _clone_from_tensor_or_other(y).float(),
    )
    y_ = y.argmax(dim=1) if len(y.shape) > 1 else y
    return (
        ce_with_logits(y_hat, y, num_classes, reduce),
        F.cross_entropy(input=y_hat.type(torch.float64), target=y, reduction=reduce),
        F.nll_loss(
            input=torch.log_softmax(y_hat.type(torch.float64), dim=1),
            target=y_,
            reduction=reduce,
        ),
    )


def linear_decay(
    a_0: float,
    a_min: float,
    t_max: float,
    t: float | Tensor,
):
    b = t / t_max
    if isinstance(t, (int, float)):
        return a_min if b > 1 else (1 - b) * a_0 + b * a_min
    return torch.where(b > 1, a_min, (1 - b) * a_0 + b * a_min)


def exp_decay(
    a_0: float,
    a_min: float,
    b: float,
    t: float | Tensor,
):
    return (a_0 - a_min) * (b**t) + a_min


def non_adaptive_step_decay(
    a_0: float,
    a_min: float,
    n: float,
    b: float,
    t: float | Tensor,
):
    t_ = int(t) if isinstance(t, (int, float)) else t.type(torch.int64)
    return (a_0 - a_min) * (b ** (t_ // n)) + a_min


def cosine_decay(
    a_0: float,
    a_min: float,
    t_max: float,
    t: float | Tensor,
):
    b = tensor(t / t_max).clone() if isinstance(t, (int, float)) else t / t_max
    return 0.5 * (a_0 - a_min) * (1 + torch.cos(b * torch.pi)) + a_min


def outsize(
    in_size: int,
    kernel_size: int,
    stride: int = None,
    padding: int = None,
    dilation: int = None,
):
    if stride == None:
        stride = DEFAULTS["stride"]
    if padding == None:
        padding = DEFAULTS["padding"]
    if dilation == None:
        dilation = DEFAULTS["dilation"]
    return math.floor(
        (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def _outsize(
    in_size: int = None,
    kernel_size: int = None,
    stride: int = DEFAULTS["stride"],
    padding: int = DEFAULTS["padding"],
    dilation: int = DEFAULTS["dilation"],
    out_size: int = None,
):
    return _solve(
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        out_size=out_size,
        eq="""
(in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1 = out_size
""",
        domain=S.Integers,
    )


def getpadding(
    in_size: int,
    kernel_size: int,
    stride: int = None,
    dilation: int = None,
    out_size: int = None,
):
    if stride == None:
        stride = DEFAULTS["stride"]
    if dilation == None:
        dilation = DEFAULTS["dilation"]
    if out_size == None:  # same conv
        out_size = in_size
    K = torch.arange(stride * (out_size - 1), stride * out_size)

    rawres = (K - in_size + dilation * (kernel_size - 1) + 1) / 2
    return rawres[rawres == rawres.type(torch.int64)].tolist()


def outsizes_seq(
    in_size: int,
    kernel_size: int | list[int],
    stride: int | list[int] = None,
    padding: int | list[int] = None,
    dilation: int | list[int] = None,
):
    _type = type(kernel_size)
    if _type == int:
        return outsize(
            in_size,
            kernel_size,
            stride,
            padding,
            dilation,
        )
    else:
        if stride == None:
            stride = [DEFAULTS["stride"]] * len(kernel_size)
        if padding == None:
            padding = [DEFAULTS["padding"]] * len(kernel_size)
        if dilation == None:
            dilation = [DEFAULTS["dilation"]] * len(kernel_size)
        sizes = [in_size]
        in_size_tmp = in_size
        for args in zip(kernel_size, stride, padding, dilation):
            in_size_tmp = outsize(in_size_tmp, *args)
            sizes.append(in_size_tmp)
        return sizes


def print_declarative(
    exp1: str | tuple, exp2: int | str | tuple, exp3: int | tuple = None
):
    if exp3 == None:
        print(f"{exp1}: {exp2}", end="")
    else:
        print(f"{exp1} = {exp2}: {exp3}", end="")
    if isinstance(exp3, tuple):
        num_elems = prod(exp3)
        print(f", {num_elems}")
        return num_elems
    else:
        print()


def attention_debug(
    qkv_sizes: tuple[int, int, int] | tuple[tuple[int, int, int], tuple[int, int, int]],
    qkv_sizes_str: (
        tuple[str, str, str] | tuple[tuple[str, str, str], tuple[str, str, str]]
    ),
    num_of_attention_heads: int,
    mask=False,
):
    q_size, k_size, v_size, q_size_str, k_size_str, v_size_str = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    _type = None
    if len(qkv_sizes) == 3:
        _type = "self"
        q_size = k_size = v_size = qkv_sizes
        q_size_str = k_size_str = v_size_str = qkv_sizes_str
    elif len(qkv_sizes) == 2:
        _type = "enc-dec"
        q_size, k_size = qkv_sizes
        v_size = k_size
        q_size_str, k_size_str = qkv_sizes_str
        v_size_str = k_size_str
    else:
        raise ValueError("Invalid attention type")

    print(f"\n{_type}-attention layer:")
    print_declarative("query_size", q_size_str, q_size)
    print_declarative("key_size", k_size_str, k_size)
    print_declarative("value_size", v_size_str, v_size)
    if (
        len(q_size) != 3
        or len(k_size) != 3
        or len(v_size) != 3
        or len(q_size_str) != 3
        or len(k_size_str) != 3
        or len(v_size_str) != 3
    ):
        raise ValueError(f"Lengths should be 3")
    if not (
        q_size[2] == k_size[2]
        and k_size[2] == v_size[2]
        and q_size_str[2] == k_size_str[2]
        and k_size_str[2] == v_size_str[2]
    ):
        raise ValueError(f"Last dims should be equal")

    embedding_size: int = q_size[2]
    embedding_size_str: int = q_size_str[2]

    embedding_size_per_head = embedding_size // num_of_attention_heads
    embedding_size_per_head_str = "embedding_size_per_head"
    num_of_attention_heads_str = "num_of_attention_heads"

    print_declarative(
        embedding_size_per_head_str,
        f"{embedding_size_str} // num_of_attention_heads",
        embedding_size_per_head,
    )

    multi_q_size = (num_of_attention_heads, *q_size[:2], embedding_size_per_head)
    multi_k_size = (num_of_attention_heads, *k_size[:2], embedding_size_per_head)
    multi_v_size = (num_of_attention_heads, *v_size[:2], embedding_size_per_head)
    multi_q_size_str = (
        num_of_attention_heads_str,
        *q_size_str[:2],
        embedding_size_per_head_str,
    )
    multi_k_size_str = (
        num_of_attention_heads_str,
        *k_size_str[:2],
        embedding_size_per_head_str,
    )
    multi_v_size_str = (
        num_of_attention_heads_str,
        *v_size_str[:2],
        embedding_size_per_head_str,
    )

    print_declarative("multi_head_query_sizes", multi_q_size_str, multi_q_size)
    print_declarative("multi_head_key_sizes", multi_k_size_str, multi_k_size)
    print_declarative("multi_head_value_sizes", multi_v_size_str, multi_v_size)

    q_max_length = multi_q_size[2]
    k_max_length = multi_k_size[2]
    dotp_size = (*multi_q_size[:2], q_max_length, k_max_length)

    q_max_length_str = multi_q_size_str[2]
    k_max_length_str = multi_k_size_str[2]
    dotp_size_str = (*multi_q_size_str[:2], q_max_length_str, k_max_length_str)

    print_declarative(
        "similarity_size (size of softmax(QK^T))", dotp_size_str, dotp_size
    )
    if mask:
        if q_max_length != k_max_length:
            raise ValueError(
                "Lengths must be equal, values entered wrong or not in self attention"
            )
        q_max_length_str = multi_q_size_str[2]
        num_head = multi_q_size[0]
        num_head_str = multi_q_size_str[0]
        num_batches = multi_q_size[1]
        num_batches_str = multi_q_size_str[1]
        num_masked_head = q_max_length * (q_max_length - 1) // 2
        num_unmasked_head = q_max_length * (q_max_length + 1) // 2

        num_masked = num_masked_head * num_head * num_batches
        num_masked_str = f"({q_max_length_str} * ({q_max_length_str} - 1) // 2) * {num_head_str} * {num_batches_str}"
        num_unmasked = num_unmasked_head * num_head * num_batches
        num_unmasked_str = f"({q_max_length_str} * ({q_max_length_str} + 1) // 2) * {num_head_str} * {num_batches_str}"

        print_declarative(
            "num_masked_self_attention_blocks",
            num_masked_str,
            num_masked,
        )
        print_declarative(
            "num_unmasked_self_attention_blocks",
            num_unmasked_str,
            num_unmasked,
        )

    weighted_sum_size = (*dotp_size[:3], multi_k_size[3])
    weighted_sum_size_str = (*dotp_size_str[:3], multi_k_size_str[3])
    print_declarative(
        "weighted_sum_size (size of softmax(QK^T)V, same as multi query)",
        weighted_sum_size_str,
        weighted_sum_size,
    )
    size_after_stack = (
        *weighted_sum_size[1:3],
        weighted_sum_size[0] * weighted_sum_size[3],
    )
    size_after_stack_str = (*weighted_sum_size_str[1:3], embedding_size_str)

    print_declarative(
        "output_size (after stack, same as query)",
        size_after_stack_str,
        size_after_stack,
    )

    return size_after_stack, size_after_stack_str


def self_attention_debug(
    input_size: tuple[int, int, int],
    input_size_str: tuple[str, str, str],
    num_of_attention_heads: int,
    mask=False,
):
    return attention_debug(
        input_size,
        input_size_str,
        num_of_attention_heads,
        mask,
    )


def enc_dec_attention_debug(
    enc_input_size: tuple[int, int, int],
    dec_input_size: tuple[int, int, int],
    enc_input_size_str: tuple[str, str, str],
    dec_input_size_str: tuple[str, str, str],
    num_of_attention_heads: int,
):
    return attention_debug(
        (dec_input_size, enc_input_size),
        (dec_input_size_str, enc_input_size_str),
        num_of_attention_heads,
    )


def transformer_debug(
    max_input_length: int | None = None,
    max_output_length: int | None = None,
    input_vocab_size: int | None = None,
    output_vocab_size: int | None = None,
    embedding_size: int | None = None,
    embedding_size_per_head: int | None = None,
    num_of_attention_heads: int | None = 1,
    num_of_encoder_blocks: int | None = None,
    num_of_decoder_blocks: int | None = None,
    num_batches: int = 1,
):
    if embedding_size_per_head == None:
        embedding_size_per_head = embedding_size // num_of_attention_heads
    num_of_encoder_blocks_str = "num_of_encoder_blocks"
    num_of_decoder_blocks_str = "num_of_decoder_blocks"

    num_skip_connection_encoders_str = "num_skip_connections_encoders"
    num_skip_connection_decoders_str = "num_skip_connections_decoders"
    num_self_att_blocks_encoder_str = "num_self_att_blocks_encoder (unmasked)"
    num_att_blocks_encoder_str = "num_att_blocks_encoder"
    num_self_att_blocks_decoder_str = "num_self_att_blocks_decoder (masked)"
    num_enc_dec_att_blocks_decoder_str = "num_enc_dec_att_blocks_decoder"
    num_att_blocks_decoder_str = "num_att_blocks_decoder"
    num_skip_connections_encoders = num_skip_connections_decoders = None
    if num_of_encoder_blocks != None:
        num_skip_connections_encoders = 2 * num_of_encoder_blocks
        print_declarative(
            num_skip_connection_encoders_str,
            f"2 * {num_of_encoder_blocks_str}",
            num_skip_connections_encoders,
        )
        print_declarative(
            num_self_att_blocks_encoder_str,
            f"{num_att_blocks_encoder_str} = {num_of_encoder_blocks_str}",
            num_of_encoder_blocks,
        )
    if num_of_decoder_blocks != None:
        num_skip_connections_decoders = 3 * num_of_decoder_blocks
        print_declarative(
            num_skip_connection_decoders_str,
            f"3 * {num_of_decoder_blocks_str}",
            num_skip_connections_decoders,
        )
        print_declarative(
            num_self_att_blocks_decoder_str,
            num_of_decoder_blocks_str,
            num_of_decoder_blocks,
        )
        print_declarative(
            num_enc_dec_att_blocks_decoder_str,
            num_of_decoder_blocks_str,
            num_of_decoder_blocks,
        )
        print_declarative(
            num_att_blocks_decoder_str,
            f"2 * {num_of_decoder_blocks_str}",
            2 * num_of_decoder_blocks,
        )
    if num_of_encoder_blocks != None and num_of_decoder_blocks != None:
        print_declarative(
            "num_skip_connections",
            f"{num_skip_connection_encoders_str} + {num_skip_connection_decoders_str}",
            num_skip_connections_encoders + num_skip_connections_decoders,
        )
        print_declarative(
            "num_self_att_blocks",
            f"{num_self_att_blocks_encoder_str} + {num_self_att_blocks_decoder_str}",
            num_of_encoder_blocks + num_of_decoder_blocks,
        )
        print_declarative(
            "num_att_blocks",
            f"{num_self_att_blocks_encoder_str} + 2 * {num_self_att_blocks_decoder_str}",
            num_of_encoder_blocks + 2 * num_of_decoder_blocks,
        )

    if input_vocab_size != None and max_input_length != None:
        input_size = (num_batches, max_input_length, input_vocab_size)
        input_size_str = (
            "batch_size",
            "max_input_length",
            "input_vocab_size",
        )
        print_declarative(
            "input_size",
            input_size_str,
            input_size,
        )
    size_w = (embedding_size, embedding_size)
    print("\nEncoder")
    print("\nSelfAttention")
    enc_self_att = SelfAttention(
        num_heads=num_of_attention_heads,
        size_Q=size_w,
        size_K=size_w,
        size_V=size_w,
        size_O=size_w,
    )
    enc_self_att(size=(num_batches, max_input_length, embedding_size))
    print("\nDecoder")
    print("\nSelfAttention")
    dec_self_att = SelfAttention(
        num_heads=num_of_attention_heads,
        size_Q=size_w,
        size_K=size_w,
        size_V=size_w,
        size_O=size_w,
    )
    dec_self_att(
        size=(num_batches, max_output_length, embedding_size),
        mask=True,
    )
    print("\nEncDecAttention")
    enc_dec_att = EncDecAttention(
        num_heads=num_of_attention_heads,
        size_Q=size_w,
        size_K=size_w,
        size_V=size_w,
        size_O=size_w,
    )
    enc_dec_att(
        size_enc=(num_batches, max_input_length, embedding_size),
        size_dec=(num_batches, max_output_length, embedding_size),
    )
    out_size = (num_batches, max_output_length, output_vocab_size)
    out_size_str = ("num_batches", "max_output_length", "output_vocab_size")
    print_declarative(
        "output_size (after softmax)",
        out_size_str,
        out_size,
    )


# Similarities
class Similarity:
    @staticmethod
    def scaled(
        u: Sequence | Tensor,
        v: Sequence | Tensor,
        scaling_factor: int = 1,
    ):
        u = _clone_from_tensor_or_other(u)
        v = _clone_from_tensor_or_other(v)
        return (u @ v) / scaling_factor

    @staticmethod
    def normal(
        u: Sequence | Tensor,
        v: Sequence | Tensor,
    ):
        return Similarity.scaled(u, v)

    @staticmethod
    def generalized(
        u: Sequence | Tensor,
        v: Sequence | Tensor,
        W: Sequence | Tensor,
    ):
        u = _clone_from_tensor_or_other(u)
        v = _clone_from_tensor_or_other(v)
        W = _clone_from_tensor_or_other(W)
        return u @ W @ v

    @staticmethod
    def cosine(
        u: Sequence | Tensor,
        v: Sequence | Tensor,
        scaling_factor: int = 1,
    ):
        u = _clone_from_tensor_or_other(u)
        v = _clone_from_tensor_or_other(v)
        return Similarity.scaled(
            u,
            v,
            torch.linalg.vector_norm(u) * torch.linalg.vector_norm(v) * scaling_factor,
        )


def _auto_tensor(
    size: Sequence | int | float,
    dtype=torch.float32,
    max_val_long: int = None,
):
    if dtype == torch.long or dtype == torch.int64 or dtype == int:
        if max_val_long == None:
            raise ValueError(f"max_val_long cannot be None")
        return torch.randint(0, max_val_long, size)
    if isinstance(size, (tuple, list)):
        return torch.randn(size, dtype=dtype)
        return torch.normal(mean=torch.zeros(size), std=1e-2, dtype=dtype)
    if isinstance(size, (int, float)):
        return torch.eye(size, dtype=dtype)


def _compensate_dim(_tensor: Tensor, target_dim: int = 2):
    curr_dim = len(_tensor.shape)
    if target_dim < curr_dim:
        return _tensor
    new_inp = _tensor
    for _ in range(target_dim - curr_dim):
        new_inp = new_inp.unsqueeze(0)
    return new_inp


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(
        self,
        num_heads=1,
        size_Q=None,
        size_K=None,
        size_V=None,
        size_O=None,
        W_query: Sequence | Tensor = None,
        W_key: Sequence | Tensor = None,
        W_value: Sequence | Tensor = None,
        W_output: Sequence | Tensor = None,
        dotp_func: str = Similarity.scaled,
        generalized_dotp_weight: Sequence | Tensor = None,
        size_G=None,
        print_tensors=False,
    ):
        super().__init__()
        self.print_tensors = print_tensors
        self.dotp_func = dotp_func
        self.num_heads = num_heads
        self.generalized_dotp_weight = generalized_dotp_weight

        W_query = _set_tensor_size(W_query, size_Q)
        W_key = _set_tensor_size(W_key, size_K)
        W_value = _set_tensor_size(W_value, size_V)
        W_output = _set_tensor_size(W_output, size_O)

        if self.dotp_func == Similarity.generalized:
            self.generalized_dotp_weight = _set_tensor_size(
                generalized_dotp_weight, size_G
            )

        self.W_query = _compensate_dim(W_query)
        self.W_key = _compensate_dim(W_key)
        self.W_value = _compensate_dim(W_value)
        self.W_output = _compensate_dim(W_output)

        self.W_query = nn.Parameter(self.W_query)
        self.W_key = nn.Parameter(self.W_key)
        self.W_value = nn.Parameter(self.W_value)
        self.W_output = nn.Parameter(self.W_output)

        # Size controls
        if self.W_query.shape[1] != self.W_key.shape[1]:
            raise ValueError("last dims of Wq and Wk dont match")

        if self.W_value.shape[1] != self.W_output.shape[0]:
            raise ValueError("last dim of Wv and first dim of Wo dont match")

        self.embed_dim = self.W_query.shape[0]
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.dim_Q = self.W_query.shape[1]
        self.dim_V = self.W_value.shape[1]
        self.d_head_Q = self.dim_Q // self.num_heads
        self.d_head_V = self.dim_V // self.num_heads

        if self.generalized_dotp_weight != None and (
            self.generalized_dotp_weight.shape[0] != self.d_head_Q
            or self.generalized_dotp_weight.shape[1] != self.d_head_Q
        ):
            raise ValueError(
                f"generalized_dotp_weight has invalid size {self.generalized_dotp_weight.shape}"
            )

    def forward(
        self,
        query: list | Tensor = None,
        key: list | Tensor = None,
        value: list | Tensor = None,
        size_Q=None,
        size_K=None,
        size_V=None,
        mask=False,
    ):
        query = _set_tensor_size(query, size_Q)
        key = _set_tensor_size(key, size_K)
        value = _set_tensor_size(value, size_V)

        _print_size_numel_tensor(query, "query", self.print_tensors)
        _print_size_numel_tensor(key, "key", self.print_tensors)
        _print_size_numel_tensor(value, "value", self.print_tensors)

        _print_size_numel_tensor(self.W_query.data, "W_query", self.print_tensors)
        _print_size_numel_tensor(self.W_key.data, "W_key", self.print_tensors)
        _print_size_numel_tensor(self.W_value.data, "W_value", self.print_tensors)
        _print_size_numel_tensor(self.W_output.data, "W_output", self.print_tensors)
        if self.generalized_dotp_weight != None:
            _print_size_numel_tensor(
                self.generalized_dotp_weight,
                "generalized_dotp_weight",
                self.print_tensors,
            )

        # Size controls
        if key.shape[-2] != value.shape[-2]:
            raise ValueError("second last dims of key and value dont match")

        if query.shape[-1] != self.W_output.shape[1]:
            raise ValueError("last dims of query and W_output dont match")

        if query.shape[-1] != self.W_query.shape[0]:
            raise ValueError("shapes for matmul dont match")

        if key.shape[-1] != self.W_key.shape[0]:
            raise ValueError("shapes for matmul dont match")

        if value.shape[-1] != self.W_value.shape[0]:
            raise ValueError("shapes for matmul dont match")

        multi_W_query = torch.stack(
            self.W_query.split(split_size=self.d_head_Q, dim=-1), dim=0
        )
        _print_size_numel_tensor(multi_W_query, "multi_W_query", self.print_tensors)
        multi_W_key = torch.stack(
            self.W_key.split(split_size=self.d_head_Q, dim=-1), dim=0
        )
        _print_size_numel_tensor(multi_W_key, "multi_W_key", self.print_tensors)
        multi_W_value = torch.stack(
            self.W_value.split(split_size=self.d_head_V, dim=-1), dim=0
        )
        _print_size_numel_tensor(multi_W_value, "multi_W_value", self.print_tensors)

        multi_query = torch.matmul(query, multi_W_query)
        _print_size_numel_tensor(multi_query, "multi_query", self.print_tensors)
        multi_key = torch.matmul(key, multi_W_key)
        _print_size_numel_tensor(multi_key, "multi_key", self.print_tensors)
        multi_value = torch.matmul(value, multi_W_value)
        _print_size_numel_tensor(multi_value, "multi_value", self.print_tensors)

        # dotp has dim (num_heads, 1, max_len, max_len)
        dotp_type = self.dotp_func.__name__
        args = [multi_query, multi_key.transpose(-2, -1)]
        if dotp_type == "scaled":
            scaling_factor = math.sqrt(multi_query.shape[-1])
            _print_dic({"scaling_factor": scaling_factor})
            args.append(scaling_factor)
        elif dotp_type == "generalized":
            args.append(self.generalized_dotp_weight)
            if self.print_tensors:
                _print_dic({"generalized_dotp_weight": self.generalized_dotp_weight})
        dotp = self.dotp_func(*args)
        _print_size_numel_tensor(dotp, "dotp", self.print_tensors)

        # attention weights has dim (num_heads, 1, max_len, max_len)
        attention_weights = F.softmax(dotp, dim=-1)
        _print_size_numel_tensor(
            attention_weights, "attention_weights", self.print_tensors
        )

        def _mask_func(input: Tensor):
            out = input.tril()
            return out / out.sum(dim=-1, keepdim=True)

        if mask:
            attention_weights = _mask_func(attention_weights)
            _print_size_numel_tensor(
                attention_weights, "attention_weights (masked)", self.print_tensors
            )

        # sum of values weighted by the attention weights for all attention heads of
        # dimensionality (num_heads, 1, max_len, d_head)
        weighted_sum = torch.matmul(attention_weights, multi_value)
        # Tuple of weighted sums for each attention head (total length = num_heads)
        # each entry has dimensionality (1, 1, max_len, dim_head)
        weighted_sum = weighted_sum.split(1, dim=0)
        # All weighted sums stacked together for efficiency this gives a torch.tensor
        # with dimensionality (max_len, num_heads * dim_head) which is equal to (max_len, embed_dim)
        weighted_sum = torch.cat(weighted_sum, dim=-1).squeeze(dim=(0, 1))
        _print_size_numel_tensor(weighted_sum, "weighted_sum", self.print_tensors)
        # Calculate output of multi-head attention module before layer normalization
        multi_head = torch.matmul(weighted_sum, self.W_output)
        _print_size_numel_tensor(multi_head, "output of attention", self.print_tensors)
        # Sum multi-head attention plus query
        output = multi_head + query
        # Pass through layer normalization
        output = self.layer_norm(output)
        # Return output and attention weights
        return output, attention_weights


class SelfAttention(MultiHeadAttention):
    def forward(
        self,
        x: list | Tensor = None,
        size=None,
        mask=False,
    ):
        return super().forward(
            x,
            x,
            x,
            size,
            size,
            size,
            mask,
        )


class EncDecAttention(MultiHeadAttention):
    def forward(
        self,
        enc_inp: list | Tensor = None,
        dec_inp: list | Tensor = None,
        size_enc=None,
        size_dec=None,
    ):
        return super().forward(
            dec_inp,
            enc_inp,
            enc_inp,
            size_dec,
            size_enc,
            size_enc,
            False,
        )


def hp_config_iter(
    configs: dict,
    condition: Callable[[dict], bool] = lambda _: True,
):
    keys = configs.keys()
    vals = configs.values()
    for instance in itertools.product(*vals):
        config = dict(zip(keys, instance))
        if condition(config):
            yield config


def print_hp_config_iter(
    configs: dict,
    condition: Callable[[dict], bool] = lambda _: True,
):
    for i, config in enumerate(hp_config_iter(configs, condition)):
        print(f"{i+1}) {config}")


class LSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        print_tensors=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_f = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.linear_i = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.linear_c = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.linear_o = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.print_tensors = print_tensors

    def forward(
        self,
        x: Sequence | Tensor = None,
        hx: Tuple[Sequence | Tensor, Sequence | Tensor] = None,
        size_x=None,
    ) -> Tuple[Sequence | Tensor, Sequence | Tensor]:

        x = _set_tensor_size(x, size_x)

        if hx is None:
            hx = self._init_hidden_state(x)

        current_hidden_state, current_cell_state = hx

        current_hidden_state = _clone_from_tensor_or_other(current_hidden_state)
        current_cell_state = _clone_from_tensor_or_other(current_cell_state)

        if x.shape[-1] != self.input_size:
            raise ValueError("Input sizes dont match")
        if current_hidden_state.shape[-1] != self.hidden_size:
            raise ValueError("Hidden sizes dont match")

        _print_size_numel_tensor(
            current_cell_state,
            "current_cell_state",
            self.print_tensors,
        )
        _print_size_numel_tensor(
            current_hidden_state,
            "current_hidden_state",
            self.print_tensors,
        )

        xh: Tensor = torch.cat((x, current_hidden_state), 1)
        _print_size_numel_tensor(
            xh,
            "xh-concated",
            self.print_tensors,
        )
        candidate_cell_state: Tensor = self.linear_c(xh)
        _print_size_numel_tensor(
            candidate_cell_state,
            "candidate_cell_state",
            self.print_tensors,
        )
        forget_gate: Tensor = self.linear_f(xh)
        _print_size_numel_tensor(
            forget_gate,
            "forget_gate",
            self.print_tensors,
        )
        input_gate: Tensor = self.linear_i(xh)
        _print_size_numel_tensor(
            input_gate,
            "input_gate",
            self.print_tensors,
        )
        output_gate: Tensor = self.linear_o(xh)
        _print_size_numel_tensor(
            output_gate,
            "output_gate",
            self.print_tensors,
        )

        candidate_cell_state = candidate_cell_state.tanh()
        _print_size_numel_tensor(
            candidate_cell_state,
            "candidate_cell_state (after tanh)",
            self.print_tensors,
        )
        forget_gate = forget_gate.sigmoid()
        _print_size_numel_tensor(
            forget_gate, "forget_gate (after sigmoid)", self.print_tensors
        )
        input_gate = input_gate.sigmoid()
        _print_size_numel_tensor(
            input_gate, "input_gate (after sigmoid)", self.print_tensors
        )
        output_gate = output_gate.sigmoid()
        _print_size_numel_tensor(
            output_gate, "output_gate (after sigmoid)", self.print_tensors
        )

        cell_state_forgot = forget_gate * current_cell_state
        _print_size_numel_tensor(
            cell_state_forgot,
            "cell_state_forgot = forget_gate * current_cell_state",
            self.print_tensors,
        )
        cell_state_add = input_gate * candidate_cell_state
        _print_size_numel_tensor(
            cell_state_add,
            "cell_state_add = input_gate * candidate_cell_state",
            self.print_tensors,
        )

        new_cell_state = cell_state_forgot + cell_state_add
        _print_size_numel_tensor(
            new_cell_state,
            "new_cell_state",
            self.print_tensors,
        )
        new_cell_state = new_cell_state.tanh()
        _print_size_numel_tensor(
            new_cell_state,
            "new_cell_state (after tanh)",
            self.print_tensors,
        )

        new_hidden_state = new_cell_state * output_gate
        _print_size_numel_tensor(
            new_hidden_state,
            "new_hidden_state = new_cell_state * output_gate",
            self.print_tensors,
        )

        return new_hidden_state, new_cell_state

    def _init_hidden_state(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        initial_hidden_state = torch.zeros(x.shape[0], self.hidden_size)
        initial_cell_state = torch.zeros(x.shape[0], self.hidden_size)
        return initial_hidden_state, initial_cell_state
