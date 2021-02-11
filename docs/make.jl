include("../src/MinimaxAdaptiveControl.jl")
using MinimaxAdaptiveControl, Documenter
makedocs(sitename="Minimax Adaptive Control", modules=[MinimaxAdaptiveControl])
deploydocs(repo = "github.com/kjellqvist/MinimaxAdaptiveControl.jl.git")
