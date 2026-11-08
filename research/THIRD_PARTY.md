DNSMOS scoring follows Microsoft's non-personalized P.835 reference inference:

- Repository: https://github.com/microsoft/DNS-Challenge
- Pinned commit: `591184a9fcb2cbdec02520fed81a32bbbf9d73ff`
- Reference: `DNSMOS/dnsmos_local.py`
- Model: `DNSMOS/DNSMOS/sig_bak_ovr.onnx`
- Model SHA-256: `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd`
- Code license: [MIT / LICENSE-CODE](https://github.com/microsoft/DNS-Challenge/blob/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/LICENSE-CODE)
- Repository data license: [CC BY 4.0 / LICENSE](https://github.com/microsoft/DNS-Challenge/blob/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/LICENSE). Retain attribution when using the fetched model and observe any additional upstream terms.

Copyright (c) Microsoft Corporation. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The P.835 calibration coefficients and short-clip repetition procedure in
`quality.py` follow this reference. Resampling uses scipy rather than librosa.
Model weights are fetched from the pinned source and are not stored in Git.

Silero VAD is bundled with the MIT-licensed faster-whisper package. Whisper
large-v3 uses the published Systran conversion pinned in the configuration;
observe the original model and dependency licenses.
