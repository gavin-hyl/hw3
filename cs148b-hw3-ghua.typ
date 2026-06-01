// Document Setup
#let title = "CS 148B HW 3 Responses"
#let author = "Gavin Hua"

// Page setup with header
#set page(
  numbering: "1",
  number-align: right,
  header: [
    #smallcaps([#title])
    #h(1fr) #smallcaps([#author])
    #line(length: 100%)
    #v(-10pt)
    #line(length:100%)
  ]
)

// Text formatting
#set par(justify: true)
#set text(
  font: "TeX Gyre Pagella",
  size: 11pt,
)

= 2 — Vision Transformer

== vit_pooling
For spatial tasks (counting, OCR, region-referring VQA), *attention pooling* — or better yet, *no pooling at all*, i.e. passing the full set of patch tokens to the decoder — should perform best, because it preserves per-patch, spatially-localized features and lets the language model attend selectively to whichever patches contain the relevant object, character, or region. Both the #raw("[CLS]") vector and mean-pooling collapse the $N$ patch tokens into a *single global vector*, which destroys the spatial layout and the ability to isolate individual regions: mean-pooling averages distinct objects together, and #raw("[CLS]") is a learned summary optimized (e.g. via CLIP's image-text objective) toward holistic, whole-image semantics rather than fine positional detail. Consequently a single CLS vector loses where things are, how many there are, and the local high-frequency content (small text glyphs, exact object positions/relations) that counting and OCR depend on — information that cannot be recovered downstream once the image is bottlenecked into one token. Attention pooling is a reasonable middle ground (a learned query can route to salient regions), but for the hardest spatial tasks keeping the full patch grid as visual tokens gives the decoder the richest signal.

== vit_patch_size
*(1) Number of patches and compute scaling.* With a $224 times 224$ image and patch size $P$,
$
  N(P) = (224 \/ P)^2 quad => quad N(8) = 784, quad N(16) = 196, quad N(32) = 49.
$
Self-attention costs $O(N^2 d_"model")$ per layer, and $N = (224\/P)^2$, so the attention cost scales as
$O(P^(-4) d_"model")$ per layer (and linearly in the number of layers). Halving $P$ multiplies the
attention cost by $approx 16$.

*(2) Measured forward-pass time.* ViT with $d_"model" = 384$, #raw("num_heads")$= 6$,
#raw("num_blocks")$= 6$, batch of 16 images at $224 times 224$. Times below are measured on CPU (mean
$plus.minus$ population std over 5–10 timed iterations after warmup); absolute numbers will be much smaller
on an A100, but the super-linear growth as $P$ shrinks is the same — #emph[[run]] re-measure on GPU with
#raw("torch.cuda.synchronize()") around the timing block and 20 iterations after 5 warmups.

#table(
  columns: 4,
  table.header([*Patch size $P$*], [*Patches $N$*], [*Mean (ms)*], [*Std (ms)*]),
  [32], [49],  [46.0],   [0.3],
  [16], [196], [166.9],  [0.2],
  [8],  [784], [1177.3], [4.8],
)
Going $P = 16 -> 8$ ($N$: $196 -> 784$) raises the time $approx 7 times$, as the $O(N^2)$ attention term
starts to dominate the $O(N d^2)$ projection/MLP terms.

*(3) When to accept the smaller-patch trade-off.* Accept the smaller-patch (more expensive) trade-off when the task is *detail- or spatially-sensitive* — small objects, fine textures, OCR/document reading, counting, or dense per-pixel prediction (segmentation/depth) — where the extra spatial resolution from more tokens $N = (224\/P)^2$ raises accuracy enough to justify the quartic $tilde P^(-4)$ growth in attention cost $O(N^2 dot d)$, *and* when compute/latency/memory is not the binding constraint (e.g., offline or accuracy-critical settings rather than real-time, edge, or large-batch serving).

= 3 — CLIP-Style Contrastive Pretraining

== infonce (Symmetric InfoNCE)
Stack a batch of $B$ image embeddings into rows of $I in RR^(B times d)$ and the $B$ text embeddings into rows of $T in RR^(B times d)$, both L2-normalized, and form the similarity matrix
$
S = I T^T exp("logit_scale"), quad S_(i j) = exp("logit_scale") dot chevron.l I_i, T_j chevron.r .
$
With targets $y = "arange"(B)$ (so the correct match for index $i$ is index $i$), the matched image-caption pairs sit on the diagonal $S_(i i)$, and the symmetric loss is
$
L = 1/2 ( "CE"(S, y) + "CE"(S^T, y) ) .
$

*The two terms are the two retrieval directions.* Reading $S$ along its rows, row $i$ is a distribution over all $B$ captions for image $i$:
$
"CE"(S, y) = -1/B sum_(i=1)^B log (exp(S_(i i)) / (sum_(j=1)^B exp(S_(i j)))) .
$
This is the *image$arrow.r$text* term: each image (row) must pick out its own caption from all captions in the batch, treating the other $B-1$ captions as negatives. Transposing swaps the roles, so row $i$ of $S^T$ is column $i$ of $S$, i.e. a distribution over all $B$ images for caption $i$:
$
"CE"(S^T, y) = -1/B sum_(i=1)^B log (exp(S_(i i)) / (sum_(j=1)^B exp(S_(j i)))) .
$
This is the *text$arrow.r$image* term: each caption (column) must pick out its own image, treating the other images as negatives. The two normalizations differ — one softmaxes over a row of $S$, the other over a column — so the terms are genuinely distinct and neither alone constrains both directions.

*Why average both.* A single term, say $"CE"(S, y)$, only makes each image's correct caption rank first *among all captions* (a row-wise ranking); it says nothing about whether each caption's correct image ranks first *among all images* (a column-wise ranking). Because these are two different rankings, optimizing one term does not imply the other, and one-directional solutions are admitted — e.g. a "hub" image embedding can sit closest to many captions without penalty from the row term, since that term never normalizes over images. Averaging the two directions makes the objective symmetric under swapping the modalities, $L(I,T) = L(T,I)$, so neither modality is privileged. It simultaneously pushes the diagonal up and pushes down *both* the off-diagonal entries within each row and within each column, which is exactly the condition that matched pairs are *mutually* nearest neighbors: caption $i$ is image $i$'s nearest text and image $i$ is caption $i$'s nearest image. This matches CLIP's deployment goals, since both retrieval directions (search images by text and search text by image) are used, so the training signal weights them equally.

== clip_train (Pretraining on EuroSAT)
Implemented in #raw("scripts/pretrain_clip.py"): the ViT + #raw("ProjectionHeads") + learnable
#raw("logit_scale") are trained with #raw("clip_loss") (AdamW, cosine schedule with warmup,
#raw("logit_scale") clamped to $log 100$); zero-shot validation accuracy is logged each epoch via
#raw("vlm.eval.zeroshot_classification_accuracy"). Hyperparameters per
#raw("configs/clip_eurosat.yaml"): lr $3 times 10^(-4)$, weight decay $0.1$, batch 256, 20 epochs, cosine
schedule with 200 warmup steps.

*(a)–(b)* #emph[[run] Paste the training-loss curve and the zero-shot validation-accuracy curve from
#raw("runs/clip_eurosat/history.json").]

*(c) How the curves relate.* Early in training the two curves move together: as the contrastive loss falls, image embeddings align with their caption embeddings and zero-shot accuracy climbs quickly. They then decouple — *zero-shot accuracy rises fast and plateaus* against a ceiling fixed by the frozen text encoder and the limited expressiveness of the class-template captions, while the *training loss can keep decreasing past this plateau* because InfoNCE keeps sharpening an already-correctly-ranked similarity matrix (pushing logits apart, effectively like a lower temperature on pairs that are already correct), which lowers the loss without changing which class wins the argmax. This decoupling is exaggerated on EuroSAT: since captions are class templates, many in-batch examples share an identical caption, so these duplicate positives are still treated as negatives in the symmetric loss and inflate its raw value, further separating the loss magnitude from the accuracy it is meant to track.

== clip_zeroshot (Qualitative analysis)
#emph[[run] Paste 5 correctly- and 5 incorrectly-classified validation images with their top-3 predicted
classes.] Discussion: classifier mistakes are expected to be #emph[reasonable] confusions between
semantically/visually adjacent land-use classes (e.g. #raw("PermanentCrop") vs #raw("HerbaceousVegetation"),
#raw("River") vs #raw("Highway")), indicating the learned embedding space groups visually similar terrain
together rather than being random — evidence that contrastive pretraining imposed meaningful structure.

= 4 — LoRA Fine-Tuning

== lora_linear (Parameter accounting)
#raw("LoRALinear") and #raw("apply_lora_to_attention") are implemented in #raw("basics/lora.py")
(#raw("test_lora_linear"), #raw("test_apply_lora") pass). For the CLIP ViT
($d_"model"=384$, 6 heads, 6 blocks) with LoRA on every head's #raw("q_proj")/#raw("v_proj") at rank $r=8$,
$alpha=16$:
#table(
  columns: 3,
  table.header([*Total params*], [*Trainable (LoRA) params*], [*Ratio*]),
  [10{,}737{,}792], [258{,}048], [2.40%],
)
Only $2.40%$ of the ViT's parameters are trained — LoRA adds $2 r d$ params per wrapped projection
($r d_"in"$ for $A$ plus $d_"out" r$ for $B$).

== lora_compare (Full FT vs LoRA vs Linear Probe)
Implemented in #raw("scripts/finetune_resisc.py") (loads the CLIP ViT, attaches a 45-way head, applies the
chosen adaptation, trains 10 epochs, dumps #raw("metrics.json") with test accuracy, trainable params, peak
memory, and wall-clock).
#emph[[run] Fill the table from the three #raw("metrics.json") files.]
#table(
  columns: 5,
  table.header([*Method*], [*Test acc*], [*Trainable params*], [*Peak mem (MB)*], [*Wall-clock (s)*]),
  [Linear probe], [_[run]_], [_[run]_], [_[run]_], [_[run]_],
  [LoRA (r=8)],   [_[run]_], [258{,}048 + head], [_[run]_], [_[run]_],
  [Full FT],      [_[run]_], [_[run]_], [_[run]_], [_[run]_],
)
Across the four axes the three methods form a consistent ordering, because each axis is governed by *how many parameters carry gradients and optimizer state*.

*Trainable parameter count (linear probe $<$ LoRA $<$ full FT).* The linear probe trains only the 45-way classification head (a single $d times 45$ matrix plus bias), so it has by far the fewest trainable parameters; LoRA adds two low-rank factors $A in RR^(r times d_("in"))$ and $B in RR^(d_("out") times r)$ per adapted projection (here #raw("q_proj") and #raw("v_proj") at rank $r = 8$, $alpha = 16$), giving roughly $2$--$3%$ of the model trainable (we measured $2.40%$); full fine-tuning makes every weight trainable ($100%$).

*Peak GPU memory (linear probe $<$ LoRA $<$ full FT).* Memory is dominated by optimizer state (Adam stores two moments per *trainable* parameter) plus the activations cached for backprop. The linear probe can pre-extract frozen features once and only backprops through the head, so it stores neither backbone activations nor backbone optimizer state and is lowest; LoRA keeps the base $W$ frozen (no gradients or moments for it) and only the tiny adapters carry optimizer state, which removes most of full FT's optimizer/gradient memory, though it must still backprop through the frozen layers and so caches activations comparable to full FT; full FT additionally stores gradients and Adam moments for every weight, so it is highest.

*Wall-clock time (linear probe $<$ LoRA $lt.tilde$ full FT).* The linear probe is fastest since features can be extracted once and only the head is optimized; LoRA and full FT both backpropagate through the whole network, so per-step cost is similar, but LoRA's smaller optimizer footprint and typically faster convergence make it modestly quicker than full FT.

*Final test accuracy (linear probe $<$ LoRA $lt.tilde$ full FT).* Because the linear probe leaves the backbone frozen, the features are never adapted to RESISC45's aerial-imagery domain, so it is typically the least accurate; full FT adapts all features and is usually the most accurate (at the highest cost and some overfitting risk on a small dataset); LoRA recovers nearly all of full FT's accuracy by injecting low-rank updates into the attention projections, making it the best *accuracy-per-cost* trade-off. The writeup should report a $4$-column table (test accuracy, trainable params, peak memory, wall-clock time) with one measured row per method.

== lora_rank (Rank sweep)
#emph[[run] Plot test accuracy vs LoRA rank $r in {1,2,4,8,16,32,64}$ ($alpha = 2r$), 10 epochs each.]
Sweeping $r in {1,2,4,8,16,32,64}$ with $alpha = 2r$, I expect accuracy to rise steeply over the smallest ranks ($r = 1 -> 2 -> 4$) and then flatten, with *diminishing returns by $r approx 8$ to $16$*: beyond this point each doubling of $r$ roughly doubles the LoRA parameter count and trainable directions but yields little or no validation-accuracy gain, and very large $r$ (32, 64) can even nudge accuracy down through mild overfitting on RESISC45. This plateau lines up almost exactly with the *$r = 8$ or $16$* values used in practice for fine-tuning large models, which is the empirically observed sweet spot between expressiveness and parameter efficiency. The agreement supports the central LoRA hypothesis that the adaptation $Delta W = (alpha/r) B A$ has *low intrinsic rank* — the task-specific update lives in a small subspace, so a handful of directions in the #raw("q_proj")/#raw("v_proj") updates capture essentially all of the useful adaptation. Consequently, adding rank past $approx 8$ to $16$ mostly grows the parameter budget without supplying new informative directions, implying the effective rank of the fine-tuning update is small even though the full weight matrices are high-dimensional.

= 5 — Vision-Language Model

#emph[The projector (#raw("vlm/projector.py")), fusion/injection/masking/label-shifting and generation
(#raw("vlm/model.py")), and the §6 #raw("return_all_tokens") flag (#raw("basics/vit.py")) are implemented;
training/eval drivers are #raw("scripts/train_vlm.py") and #raw("scripts/eval_vlm.py"). Shared QA-batch
formatting (answer-only #raw("ignore_index=-100") masking) lives in #raw("vlm/textbatch.py").]

== projector (Why a 2-layer MLP)
When the encoder and decoder are both frozen, the projector is the *only* trainable component, so it alone must carry the entire burden of aligning the visual feature manifold to the decoder's token-embedding manifold. A single linear layer can realize only an affine map $z |-> W z + b$ — a global rotation, scaling, shear, and shift applied uniformly everywhere. But the geometry the frozen encoder produces and the geometry the frozen decoder expects are generally related by a *nonlinear* warp, and one rigid affine transform cannot place clustered, manifold-structured visual features into the regions of embedding space the frozen decoder treats as meaningful token vectors.

A 2-layer MLP (#raw("Linear") $->$ #raw("GELU") $->$ #raw("Linear")) supplies the missing nonlinearity:
$
h = W_2 dot "GELU"(W_1 z + b_1) + b_2 .
$
The #raw("GELU") lets the projector bend and fold the visual space and apply different effective transforms to different feature directions, rather than the same affine map everywhere; with a wide-enough hidden layer it is a universal approximator, so it can match the frozen decoder's expected input distribution far more faithfully than a linear map. (A single linear projector still works to a degree, but worse — and once the decoder is *unfrozen*, it can absorb part of this adaptation, making a linear projector more viable; during frozen-frozen pretraining the nonlinear projector is what makes alignment expressive enough to learn well.)

== injection_compare (Which injection strategy)
#emph[[run] Train cls / all_patches / interleaved for 2000 steps (projector only, batch 32, lr $10^(-4)$)
and fill the table.]
#table(
  columns: 5,
  table.header([*Injection*], [*Val exact-match*], [*Visual tokens / ex.*], [*Peak mem (MB)*], [*Time / step (s)*]),
  [cls],          [_[run]_], [1],     [_[run]_], [_[run]_],
  [all_patches],  [_[run]_], [$N+1$], [_[run]_], [_[run]_],
  [interleaved],  [_[run]_], [$N+1$], [_[run]_], [_[run]_],
)
*Expected ordering (CLEVR exact-match):* #raw("all_patches") $approx$ #raw("interleaved") $>$ #raw("cls"). The #raw("cls") strategy injects a single global #raw("[CLS]") summary token, which is the cheapest by far (1 visual token, smallest sequence length, lowest peak memory, fastest time/step) but collapses the whole image into one vector, discarding per-region detail and spatial layout; this cripples it on CLEVR's compositional/counting/spatial questions (e.g. "the cube behind the red sphere") that require localizing and comparing specific objects. Both #raw("all_patches") and #raw("interleaved") inject the same $N+1$ visual tokens, giving the decoder direct attention access to every patch so it can route to the relevant object/region and reason about relations, yielding the best accuracy; the only difference is *where* the tokens sit in the sequence (a prefix vs. at the #raw("<image>") placeholder), not how much information is available, so their scores are close. The cost trade-off is monotone: going from #raw("cls") to the $N+1$-token variants adds $N$ extra tokens per example, which raises peak memory (KV-cache grows linearly and attention compute grows quadratically in sequence length) and slows time/step. This is exactly the #raw("[CLS]")-vs-patch pooling trade-off from the #raw("vit_pooling") question, now relocated to the decoder interface: passing only the #raw("[CLS]") token is the cheap-but-spatially-lossy global summary, whereas passing all patch tokens preserves fine-grained, position-resolved features — and just as in pooling, the extra cost is usually worth it for spatial VQA like CLEVR.

== masking (Image-block attention)
*(1) Mask diagrams* (4 visual tokens $v_1..v_4$ followed by 3 text tokens $t_1..t_3$; shaded = allowed; red
lines mark the image/text boundary):
#figure(image("assets/masking_m1_m2.png", width: 95%))
Under (M1) fully-causal, $v_i$ attends only to $v_(<= i)$ (28 allowed cells); under (M2), the visual block is
fully bidirectional (top-left $4 times 4$ all shaded) while text stays causal and attends to all visual
tokens (34 allowed cells).

*(2) Which is better, and why.* *M2 is expected to perform better.* Unlike text, an image has no natural causal or raster ordering: patch $i$ is not "before" patch $j$ in any meaningful sense, so M1's causal mask is an arbitrary constraint that forbids early patches from attending to later ones and is inconsistent with how the features were produced — the ViT encoder is *bidirectional*, building each patch from full 2D context, so a decoder that re-imposes a 1D causal order on the visual prefix is a mismatch. M2 instead keeps the masking aligned with the modalities: bidirectional inside the image block, so every visual token attends to every other and the decoder reads a representation of the *whole* image, while remaining causal across the image$->$text boundary and among the text, exactly the constraint autoregressive generation needs. Thus M2 hands the decoder a richer visual prefix without sacrificing language-modeling causality, whereas M1 only degrades the visual representation for no benefit.

*(3) Both masks, 500 steps each.* #emph[[run] Train all_patches injection with each mask for 500 steps and
report validation accuracy.]
#table(
  columns: 2,
  table.header([*Mask*], [*Val exact-match*]),
  [M1 (causal)],       [_[run]_],
  [M2 (image_bidir)],  [_[run]_],
)

== freezing (What to train, and when)
#emph[[run] Run configs A–D for 1500 steps each (best injection + mask from above) and fill the table.]
#table(
  columns: 4,
  table.header([*Config*], [*Val exact-match*], [*Trainable params*], [*Peak mem (MB)*]),
  [A: projector only],        [_[run]_], [_[run]_], [_[run]_],
  [B: + decoder LoRA (r=8)],  [_[run]_], [_[run]_], [_[run]_],
  [C: + full decoder FT],     [_[run]_], [_[run]_], [_[run]_],
  [D: all three FT],          [_[run]_], [_[run]_], [_[run]_],
)
All four runs share the same 1500-step budget but differ sharply in trainable parameters and memory. *Config A* (projector-only) is cheapest — it backpropagates into a single small MLP while the encoder and decoder stay frozen, so it only learns to align visual features into the frozen LM's token space; this makes it an excellent stage-1 alignment step but caps accuracy, because the language model cannot adapt its next-token distribution to the new visual-conditioned inputs. *Config B* (projector + decoder-LoRA) adds only the low-rank $B A$ updates ($A in RR^(r times d_("in"))$, $B in RR^(d_("out") times r)$, base $W$ frozen), which lets the decoder adapt to visual conditioning at a tiny fraction of the parameters, memory, and optimizer state of full fine-tuning, and in practice recovers most of full-FT's accuracy gain — giving the *best accuracy/cost trade-off*. *Config C* (projector + full decoder FT) can match or slightly exceed B, but it is far more expensive (full gradients, optimizer moments, and activation memory for all decoder weights) and, on a small dataset like CLEVR, risks overwriting the LM's general language ability (catastrophic forgetting) since every weight is free to drift. *Config D* additionally unfreezes the ViT encoder: this is the most expensive and the most prone to overfitting and forgetting, because the visual backbone and LM can co-adapt to a narrow task distribution and lose their pretrained generality, with little expected payoff at only 1500 steps. In the standard two-stage recipe this maps cleanly onto *pretraining then instruction-tuning*: run an A-style projector-only pass first to align vision to the frozen LM, then lightly unfreeze with B-style LoRA (optionally the encoder last, with a small learning rate) for instruction tuning — rather than doing full fine-tuning of everything at once, which wastes compute and endangers the pretrained capabilities.

== vlm_qualitative (What has the VLM learned)
#emph[[run] Generate on 10 held-out CLEVR examples (#raw("scripts/eval_vlm.py --save-images")); include
image, question, gold answer, and model generation for a mix of correct/incorrect cases.] For each error,
hypothesize encoder failure (image not understood — e.g. wrong color/shape/count) vs decoder failure
(question misread — e.g. answers a different attribute, or ignores a negation/relation). A clean experiment
to distinguish them: hold the image fixed and paraphrase the question (decoder probe), and hold the question
fixed and swap in an easier/cropped image or feed ground-truth scene attributes as text (encoder probe);
errors that move with the question are decoder-side, errors that move with the image are encoder-side.

= 6 — Positional Encodings and RoPE

== rope_1d (1D RoPE + manual norm check)
#raw("RoPE1D") and #raw("RoPE2D") are implemented in #raw("basics/rope.py") (#raw("test_rope_1d"),
#raw("test_rope_2d") pass; cos/sin tables are precomputed as non-persistent buffers).
RoPE acts on each disjoint 2D coordinate pair $(x_(2i), x_(2i+1))$ by left-multiplying it with the rotation matrix $R_theta = mat(cos theta, -sin theta; sin theta, cos theta)$, which is orthogonal ($R_theta^T R_theta = I$, determinant $1$), so each pair's Euclidean length is unchanged; since $norm(x)^2 = sum_i norm("pair"_i)^2$ is just the sum of these preserved per-pair norms, the full vector norm is preserved exactly in exact arithmetic. From the manual check I would report the measured $max_x abs(norm("RoPE"(x)) - norm(x))$ over the test vectors, which should be on the order of $approx 1 times 10^(-6)$ (i.e. pure floating-point rounding, not a real discrepancy). This norm-invariance is precisely why RoPE is applied to $q$ and $k$ but not $v$: it rewrites only the angles (relative phases) so the dot product $q^T k$ becomes a function of relative position, while leaving $norm(q)$ and $norm(k)$ -- and hence the overall scale of the attention logits -- intact.

Measured: max $|thin norm(x) - norm("RoPE"(x)) thin| = 9.54 times 10^(-7)$ (float-32 rounding), for both
1D and 2D RoPE.

*Relative-position property.* RoPE replaces each $2$-dimensional sub-vector of $q$ at position $m$ with a copy rotated by angle $m theta_i$, and the corresponding sub-vector of $k$ at position $n$ by $n theta_i$, where $theta_i = "base"^(-2i\/d)$ is the $i$-th frequency. For any two planar vectors, the inner product of $R(m theta_i) q$ with $R(n theta_i) k$ is $q^T R(m theta_i)^T R(n theta_i) k = q^T R((n - m) theta_i) k$, using $R(a)^T R(b) = R(b - a)$ — it depends on the angle difference only, with the absolute angles $m theta_i$ and $n theta_i$ cancelling.

Summing over all frequency pairs, the full dot product becomes

$ q_m dot k_n = sum_i q_i^T R((n - m) theta_i) k_i, $

a function of the offset $m - n$ alone. Hence $(3,8)$ and $(10,15)$ both have offset $5$ and yield identical dot products: attention scores carry relative-position information with no absolute-position leakage.
Measured: with $q$ at $m$ and $k$ at $n$, the dot product at offset $5$ is identical for $(m,n)=(3,8)$ and
$(10,15)$ — both $-2.5769$, differing by $4.8 times 10^(-7)$.

== rope_vs_learned (Learned PE vs 1D RoPE)
#emph[[run] Retrain CLIP on EuroSAT for 20 epochs with (a) learned PE and (b) 1D RoPE
(#raw("--pos-encoding rope_1d")); report zero-shot val accuracy, then the length-extrapolation test:
evaluate both at $96 times 96$ (144 patches, patch size 8) by interpolating the learned $8 times 8$
positional grid to $12 times 12$ for the baseline.]
#table(
  columns: 3,
  table.header([*PE method*], [*Train-size acc (64×64)*], [*Extrapolated acc (96×96)*]),
  [Learned PE], [_[run]_], [_[run]_],
  [1D RoPE],    [_[run]_], [_[run]_],
)
Expected: comparable accuracy at train resolution, but learned PE degrades sharply at $96 times 96$ (it must
interpolate a fixed grid and was never trained on those absolute positions), whereas RoPE — being relative
and analytic in position — extrapolates far more gracefully.

== rope_2d (2D RoPE for image patches)
#raw("RoPE2D") splits the head dimension in half: the first half is rotated by the patch's $x$-coordinate,
the second by its $y$-coordinate (the ViT supplies each patch's $(x,y)$ grid index;
#raw("--pos-encoding rope_2d")). #emph[[run] Re-run CLIP pretraining + zero-shot eval (and the
length-extrapolation test) with 2D RoPE.] Expected: 2D RoPE matches or beats 1D RoPE on EuroSAT because it
respects the true 2D adjacency of patches (a patch and the one directly below it are near in $y$), and it
extrapolates well for the same relative-position reason.

== mrope_written (Reasoning about M-RoPE)
*(1) Naive 1D position IDs.* If we lay out the visual tokens as a flat prefix and number everything $0,1,2,dots$, the $64$ patch tokens plus the #raw("[CLS]") consume $approx 65$ positions before the prompt even begins, so the first text token sits at position ID $65$ and the $50$-token prompt occupies indices $65 dots 114$. This breaks the decoder in two ways. *(a) Range / extrapolation:* during text-only pretraining the same prompt would have lived at positions $0 dots 49$, so every text token is now shifted $approx 65$ indices higher and the pairwise relative offsets that feed RoPE land in a regime the rotary tables (and the learned attention behavior) were never trained on; the decoder must extrapolate, degrading quality. *(b) Lost 2D structure:* a raster scan forces an artificial 1D order on a 2D grid, so two patches that are vertically adjacent in the image (one directly below the other) differ by a full row width $g$ in 1D index. RoPE then assigns them a large relative offset even though they are spatial neighbors, so the relative-position signal no longer reflects true spatial adjacency.

*(2) First text position under M-RoPE.* Each image patch receives a 3D position $(t, x, y)$ with a shared temporal index $t$ and $(x,y)$ ranging over the patch grid, i.e. $x,y in {0, dots, g-1}$ for a $g times g$ grid. Text resumes at one past the *maximum* index any coordinate of the image reached, so the first text token gets position $max(g_h, g_w)$ (for a square grid, $g$, i.e. $sqrt(N)$, rather than $N = g^2$). This is sensible because the image's coordinates only span $0 dots g-1$ along each axis: advancing text from $max + 1$ keeps the text positions compact and in-distribution (the prompt occupies roughly $g dots g+49$ instead of $65 dots 114$), avoiding the giant jump of the full patch count, while still guaranteeing the text is ordered strictly after the image.

*(3) Three chunks instead of two.* Splitting the head dimension into three groups $(t, x, y)$ lets one rotary mechanism simultaneously encode *temporal/sequential* order in the $t$ chunk and *2D spatial* position in the $x$ and $y$ chunks. The temporal axis is what supplies a linear ordering across frames of a video, across multiple images, and across the inherently 1D sequence of text tokens. If we dropped $t$ and used only $(x,y)$, text tokens (which are sequential, not spatial) and distinct frames/images would have no ordering coordinate: every text token would have to be squeezed onto the same 2D grid, and two tokens at different sequence positions, or the same patch coordinate in two different frames, would collide to identical positions. That destroys the causal/sequential structure the decoder relies on for text and for distinguishing multi-image or multi-frame inputs, which is exactly why the temporal chunk is kept.

== mrope_impl (Implementing M-RoPE — bonus)
#emph[[run] Assign M-RoPE-style 3D positions $(t,x,y)$ to the visual + text tokens and retrain the best §5
config for 1500 steps with (a) naive 1D position IDs and (b) M-RoPE IDs; report overall and spatial-question
accuracy.]
#table(
  columns: 3,
  table.header([*Position scheme*], [*Overall acc*], [*Spatial-question acc*]),
  [Naive 1D IDs], [_[run]_], [_[run]_],
  [M-RoPE IDs],   [_[run]_], [_[run]_],
)
Expected: M-RoPE helps most on spatial-relation questions (#emph["left of"], #emph["behind"],
#emph["in front of"]) because it encodes the patch grid's 2D structure and keeps text positions compact,
with a smaller effect on non-spatial (count/existence/attribute) questions.
