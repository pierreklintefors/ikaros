# AveragePool

## Description

AveragePool replaces each square window of `size` by `size` pixels with its mean, moving the window
`stride` pixels at a time. With `stride` equal to `size`, the windows tile the image; with a smaller
`stride`, neighbouring windows overlap, so something that lies across a window boundary is still
found whole in some window. A channel-first input `[channels, rows, cols]` is pooled channel by
channel. Pixels beyond the last whole window are not used.

Applied to a binary map, each output value is the fraction of the window that is set. For example,
`EpiAnimatronics.ikg` pools the maps of corn-yellow and leaf-green pixels and requires both colours
in the same window, so that the toy corn counts but a yellow object and a green object in different
places do not.

## Parameters

| Name | Default | Description |
| --- | --- | --- |
| `size` | 2 | Width and height of each window in pixels |
| `stride` | 2 | Pixels the window moves between neighbouring outputs |

## Inputs

| Name | Description |
| --- | --- |
| INPUT | Image `[rows, cols]` or channel-first image `[channels, rows, cols]` |

## Outputs

| Name | Description |
| --- | --- |
| OUTPUT | Window means, `(rows - size) / stride + 1` by `(cols - size) / stride + 1`, rounded down |
