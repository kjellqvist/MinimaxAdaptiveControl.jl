using MinimaxAdaptiveControl, Documenter
makedocs(
         sitename="Minimax Adaptive Control",
         modules=[MinimaxAdaptiveControl],
         pages=[
                "Home" => "index.md",
                "Examples" => [
                               "State Feedback: Double Integrator" => "examples/ss_double_integrator.md",
                               "State Feedback Delayed Integrator" =>"examples/ss_delayed_integrator.md",
                               "Output feedback" =>"examples/output_feedback.md"
                              ],
                "Functions" => [
                               "Constructors" => "lib/constructors.md",
                               "Reduction" => "lib/reduction.md",
                               "Bellman Inequalities" => "lib/bellman.md",
                               "Self-tuning Regulator" => "lib/str.md",
                               "Simulation" => "lib/simulation.md"
                              ],
                "API" => "api.md"
               ],
         checkdocs=:exports
        )
#deploydocs(repo = "github.com/kjellqvist/MinimaxAdaptiveControl.jl.git")
