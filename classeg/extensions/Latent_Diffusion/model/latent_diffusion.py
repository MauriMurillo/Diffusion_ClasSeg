from typing import Tuple, List

from classeg.extensions.Latent_Diffusion.utils.utils import make_zero_conv
import torch
import torch.nn as nn
from classeg.extensions.Latent_Diffusion.model.modules import ScaleULayer

class LateInitializationLayerNorm(nn.Module):
    def __init__(self, **kwargs):
        super(LateInitializationLayerNorm, self).__init__()
        self.ln = None

    def forward(self, x):
        if self.ln is None:
            if self.training:
                print(
                    "WARNING: LateInitializationLayerNorm is in training and is initializing itself now. "
                    "Concider running an input through first to complete initialization early"
                )
            self.ln = nn.LayerNorm(x.shape[1:]).to(x.device)
        return self.ln(x)


class TimeEmbedder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, t, n):
        # TODO remove arg
        return self.embedder(t)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim=100,
        non_lin=nn.SiLU,
        num_heads=4,
        num_layers=1,
        kernel_size=3,
        stride=1,
        padding=1,
        downsample=True,
        attention=False,
        apply_zero_conv=False,
        apply_scale_u=False,
    ) -> None:
        """
        TODO GroupNorm
        Residual block x -> {NORM->NonLin->Conv -> (TIME EMBEDDING)-> Norm -> NonLin -> Conv} -> x' -> {concat x -> Norm -> SA} -> Concat x' -> Down
        """
        super().__init__()
        self.perform_attention = attention

        self.apply_zero_conv = apply_zero_conv
        self.apply_scale_u = apply_scale_u
        assert (
            not self.apply_scale_u or not self.apply_zero_conv
        ), "Cannot do both scaleu and zero conv"
        if self.apply_zero_conv:
            self.zero_conv = make_zero_conv(channels=in_channels, conv_op=nn.Conv2d)
        if self.apply_scale_u:
            self.scale_u = ScaleULayer(in_channels, skipped_count=1)
            self.scale_u_pointwise_convolution = nn.Conv2d(
                in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1
            )

        self.num_layers = num_layers
        self.first_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    non_lin(),
                    nn.Conv2d(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.time_embedding_layer = nn.ModuleList(
            [TimeEmbedder(time_emb_dim, out_channels) for _ in range(num_layers)]
        )
        self.second_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    non_lin(),
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        if self.perform_attention:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(8, num_channels=out_channels) for _ in range(num_layers)]
            )
            self.multihead_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
        self.pointwise_convolution = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )
        self.downsample_conv = (
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if downsample
            else nn.Identity()
        )

    def forward(self, x, time_embedding, residual_connection=None):
        out = x
        if residual_connection is not None:
            if self.apply_zero_conv:
                out = out + self.zero_conv(residual_connection)
            elif self.apply_scale_u:
                out = self.scale_u_pointwise_convolution(
                    self.scale_u(x, residual_connection)
                )
            else:
                out = out + residual_connection

        # TODO check time embeedding domension stuff
        for layer in range(self.num_layers):
            res_input = out
            out = self.first_residual_convs[layer](out)
            out = (
                out
                + self.time_embedding_layer[layer](time_embedding, out.shape[0])[
                    :, :, None, None
                ]
            )
            out = self.second_residual_convs[layer](out)
            # Skipped connection
            out = out + self.pointwise_convolution[layer](res_input)

            if self.perform_attention:
                # Attention
                N, C, H, W = out.shape
                in_attn = out.reshape(N, C, H * W)
                in_attn = self.attention_norms[layer](in_attn).transpose(1, 2)
                out_attn, _ = self.multihead_attentions[layer](
                    in_attn, in_attn, in_attn
                )
                out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
                # Skipped
                # TODO maybe concat?
                out = out + out_attn
        out = self.downsample_conv(out)
        # print('output shape: ', out.shape)
        return out


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        time_emb_dim=100,
        conv_op="Conv2d",
        non_lin=nn.SiLU,
        num_heads=4,
        kernel_size=3,
        stride=1,
        padding=1,
    ) -> None:
        """
        TODO GroupNorm
        Residual block x -> {NORM->NonLin->Conv -> (TIME EMBEDDING)-> Norm -> NonLin -> Conv} -> x' -> {concat x -> Norm -> SA} -> Concat x' -> Down
        """
        super().__init__()
        self.first_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels),
                    non_lin(),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                )
                for _ in range(2)
            ]
        )

        self.time_embedding_layer = nn.ModuleList(
            [TimeEmbedder(time_emb_dim, in_channels) for _ in range(2)]
        )

        self.self_attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, num_channels=in_channels) for _ in range(2)]
        )
        self.self_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
                for _ in range(2)
            ]
        )
        self.cross_attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, num_channels=in_channels) for _ in range(2)]
        )
        self.cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
                for _ in range(2)
            ]
        )
        self.pointwise_convolution = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, im, seg, time_embedding):
        # ================ SELF IM ====================
        im_out = self.first_residual_convs[0](im)
        im_out = (
            im_out
            + self.time_embedding_layer[0](time_embedding, im_out.shape[0])[
                :, :, None, None
            ]
        )

        N, C, H, W = im_out.shape
        in_attn = im_out.reshape(N, C, H * W)
        in_attn = self.self_attention_norms[0](in_attn).transpose(1, 2)
        out_attn, _ = self.self_attentions[0](in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
        im_out = im_out + out_attn
        # ================ SELF SEG ====================
        seg_out = self.first_residual_convs[1](seg)
        seg_out = (
            seg_out
            + self.time_embedding_layer[1](time_embedding, seg_out.shape[1])[
                :, :, None, None
            ]
        )

        N, C, H, W = seg_out.shape
        in_attn = seg_out.reshape(N, C, H * W)
        in_attn = self.self_attention_norms[1](in_attn).transpose(1, 2)
        out_attn, _ = self.self_attentions[1](in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
        seg_out = seg_out + out_attn
        # ================ CROSS ATTENTION SEG====================
        N, C, H, W = seg_out.shape
        in_attn = seg_out.reshape(N, C, H * W)
        in_attn = self.cross_attention_norms[1](in_attn).transpose(1, 2)

        kv_attn = im_out.reshape(N, C, H * W)
        kv_attn = self.cross_attention_norms[1](kv_attn).transpose(1, 2)

        out_attn, _ = self.cross_attentions[1](in_attn, kv_attn, kv_attn)
        out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
        seg_out = seg_out + out_attn
        # ================ CROSS ATTENTION IM====================
        N, C, H, W = im_out.shape
        in_attn = seg_out.reshape(N, C, H * W)
        in_attn = self.cross_attention_norms[0](in_attn).transpose(1, 2)

        kv_attn = seg_out.reshape(N, C, H * W)
        kv_attn = self.cross_attention_norms[0](kv_attn).transpose(1, 2)

        out_attn, _ = self.cross_attentions[0](in_attn, kv_attn, kv_attn)
        out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
        im_out = im_out + out_attn

        # FINAL CONV

        return im_out, seg_out


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim=100,
        norm_op="BatchNorm",
        conv_op="Conv2d",
        non_lin=nn.SiLU,
        num_heads=4,
        num_layers=1,
        kernel_size=3,
        stride=1,
        padding=1,
        upsample=True,
        attention=False,
    ) -> None:
        """
        TODO GroupNorm
        Residual block x -> {NORM->NonLin->Conv -> (TIME EMBEDDING)-> Norm -> NonLin -> Conv} -> x' -> {concat x -> Norm -> SA} -> Concat x' -> Down
        """
        super().__init__()
        self.perform_attention = attention
        self.num_layers = num_layers
        self.upsample = upsample
        self.first_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    non_lin(),
                    nn.Conv2d(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.time_embedding_layer = nn.ModuleList(
            [TimeEmbedder(time_emb_dim, out_channels) for _ in range(num_layers)]
        )
        self.second_residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    non_lin(),
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        if self.perform_attention:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(8, num_channels=out_channels) for _ in range(num_layers)]
            )
            self.multihead_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
        self.pointwise_convolution = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )
        self.upsample_conv = (
            nn.ConvTranspose2d(
                in_channels * 3, in_channels, kernel_size=4, stride=2, padding=1
            )
            if upsample
            else nn.Identity()
        )
        self.scale_u = ScaleULayer(in_channels, skipped_count=2)

    def forward(
        self, x, skipped_connection_encoder, skipped_connection_decoder, time_embedding
    ):
        # print('UpBlock forward')
        # print('receive input x: ', x.shape)
        # print('receive skipped: ', skipped_connection.shape)
        # x = in_channels
        # x = in_channels//2
        x = self.scale_u(x, skipped_connection_encoder, skipped_connection_decoder)
        x = self.upsample_conv(x)

        out = x
        # TODO check time embeedding domension stuff
        for layer in range(self.num_layers):
            res_input = out
            out = self.first_residual_convs[layer](out)
            out = (
                out
                + self.time_embedding_layer[layer](time_embedding, out.shape[0])[
                    :, :, None, None
                ]
            )
            out = self.second_residual_convs[layer](out)
            # Skipped connection
            out = out + self.pointwise_convolution[layer](res_input)

            if self.perform_attention:
                # Attention
                N, C, H, W = out.shape
                in_attn = out.reshape(N, C, H * W)
                in_attn = self.attention_norms[layer](in_attn).transpose(1, 2)
                out_attn, _ = self.multihead_attentions[layer](
                    in_attn, in_attn, in_attn
                )
                out_attn = out_attn.transpose(1, 2).reshape(N, C, H, W)
                # Skipped
                # TODO maybe concat?
                out = out + out_attn
        # print('output shape: ', out.shape)
        return out


class LatentDiffusion(nn.Module):
    def __init__(
        self,
        lat_channels,
        layer_depth=2,
        channels=None,
        attn_channels=None,
        time_emb_dim=100,
        shared_encoder=False,
        apply_zero_conv=False,
        apply_scale_u=True,
    ):

        super(LatentDiffusion, self).__init__()
        self.apply_scale_u = apply_scale_u
        self.apply_zero_conv = apply_zero_conv

        assert (
            not shared_encoder or not apply_zero_conv
        ), "Zero conv requires separate encoders"
        self.time_emb_dim = time_emb_dim

        if channels is None:
            channels = [16, 32, 64]
        layers = len(channels)

        self.shared_encoder = shared_encoder
        self.layers = layers
        self.channels = channels

        if attn_channels is None:
            attn_channels = []
        self.attn_channels = attn_channels
        self.time_emb_dim = time_emb_dim
        self.layer_depth = layer_depth

        # Sinusoidal embedding
        self.t_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial Convolution
        self.im_conv_in = nn.Conv2d(
            in_channels=lat_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.seg_conv_in = nn.Conv2d(
            in_channels=lat_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Encoder
        self.encoder_layers = self._generate_encoder()

        if not shared_encoder:
            self.encoder_layers_mask = self._generate_encoder()

        # Middle
        mid_channels = channels[-1]
        self.middle_layer = MidBlock(
            in_channels=mid_channels,
            time_emb_dim=self.time_emb_dim,
        )
        # Decoder IM
        self.im_decoder_layers = nn.ModuleList()
        for layer in range(layers - 1, 0, -1):
            in_channels = channels[layer]
            out_channels = channels[layer - 1]
            self.im_decoder_layers.append(
                UpBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_emb_dim=self.time_emb_dim,
                    upsample=True,
                    num_layers=layer_depth,
                    attention=(in_channels in self.attn_channels)
                )
            )

        # Decoder SEG
        self.seg_decoder_layers = nn.ModuleList()
        for layer in range(layers - 1, 0, -1):
            in_channels = channels[layer]
            out_channels = channels[layer - 1]
            self.seg_decoder_layers.append(
                UpBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_emb_dim=self.time_emb_dim,
                    upsample=True,
                    num_layers=layer_depth,
                    attention=(in_channels in self.attn_channels)
                )
            )

        self.output_layer_im = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=lat_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.output_layer_seg = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=lat_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def _generate_encoder(self):
        encoder_layers = nn.ModuleList()
        for layer in range(self.layers - 1):
            # We want to build a downblock here.
            in_channels = self.channels[layer]
            out_channels = self.channels[layer + 1]
            encoder_layers.append(
                DownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_emb_dim=self.time_emb_dim,
                    downsample=True,
                    num_layers=self.layer_depth,
                    apply_zero_conv=self.apply_zero_conv,
                    apply_scale_u=self.apply_scale_u,
                    attention=(in_channels in self.attn_channels)
                )
            )
        return encoder_layers

    def _sinusoidal_embedding(self, t):
        assert self.time_emb_dim % 2 == 0
        factor = 10000 ** (
            (
                torch.arange(
                    start=0,
                    end=self.time_emb_dim // 2,
                    dtype=torch.float32,
                    device=t.device,
                )
                / (self.time_emb_dim // 2)
            )
        )
        t_emb = t[:, None].repeat(1, self.time_emb_dim // 2) / factor
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        return t_emb

    def _encode_forward(
        self, im_out, seg_out, t
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Encodes im and seg, keeping in mind shared encoder vs not.
        If no shared encoder, turns on Residual connections between the two
        :param im_out:
        :param seg_out:
        :return:
        """
        skipped_connections_im = [im_out]
        skipped_connections_seg = [seg_out]
        # =========== SHARED ENCODER ===========
        if self.shared_encoder:
            for encoder in self.encoder_layers:
                im_out = encoder(im_out, t)
                skipped_connections_im.append(im_out)
            for encoder in self.encoder_layers:
                seg_out = encoder(seg_out, t)
                skipped_connections_seg.append(seg_out)
            return im_out, seg_out, skipped_connections_im, skipped_connections_seg
        # =========== NOT SHARED ENCODER ===========
        for i, (im_encode, seg_encode) in enumerate(
            zip(self.encoder_layers, self.encoder_layers_mask)
        ):
            if i == 0:
                im_out, seg_out = im_encode(im_out, t), seg_encode(seg_out, t)
            else:
                im_out, seg_out = (
                    im_encode(im_out, t, residual_connection=seg_out),
                    seg_encode(seg_out, t, residual_connection=im_out),
                )
            skipped_connections_im.append(im_out)
            skipped_connections_seg.append(seg_out)

        return im_out, seg_out, skipped_connections_im, skipped_connections_seg

    def forward(self, im, seg, t):

        # ======== TIME ========
        t = self._sinusoidal_embedding(t)
        t = self.t_proj(t)

        # ======== ENTRY ========
        im_out = self.im_conv_in(im)
        seg_out = self.seg_conv_in(seg)
        # ======== ENCODE ========
        im_out, seg_out, skipped_connections_im, skipped_connections_seg = (
            self._encode_forward(im_out, seg_out, t)
        )
        # ======== MIDDLE ========
        im_out, seg_out = self.middle_layer(im_out, seg_out, t)
        # ======== DECODE ========
        i = 0
        for im_decode, seg_decode in zip(
            self.im_decoder_layers, self.seg_decoder_layers
        ):
            i += 1
            im_out, seg_out = (
                im_decode(im_out, skipped_connections_im[-i], seg_out, t),
                seg_decode(seg_out, skipped_connections_seg[-i], im_out, t),
            )
        # ======== EXIT ========
        im_out = self.output_layer_im(im_out)
        seg_out = self.output_layer_seg(seg_out)
        return im_out, seg_out


if __name__ == "__main__":
    torch.cuda.empty_cache()
    in_shape = 64
    im = torch.randn(1, 4, in_shape, in_shape).float().cuda(0)
    seg = torch.randn(1, 4, in_shape, in_shape).float().cuda(0)

    model = LatentDiffusion( 
        lat_channels=4,
        layer_depth=2,
        channels=[16,32,64],
        attn_channels=[16, 32,64],
        time_emb_dim=128,
        shared_encoder=False,
        apply_zero_conv=False,
        apply_scale_u=True
    ).cuda(0)

    print("---------------")
    print(model)
    print("---------------")
    all_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {all_params}")
    print(f"Trainable params: {trainable_params}")
    y_im, y_seg = model(im, seg, torch.rand((1)).cuda(0))
    print(y_im.shape)
    print(y_seg.shape)

