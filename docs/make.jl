using TimerOutputs

dto = TimerOutput()
reset_timer!(dto)

const liveserver = "liveserver" in ARGS

if liveserver
    using Revise
    @timeit dto "Revise.revise()" Revise.revise()
end

using Documenter, DocumenterCitations, FerriteMultigrid

const is_ci = haskey(ENV, "GITHUB_ACTIONS")

# Generate tutorials and how-to guides
include("generate.jl")


bibtex_plugin = CitationBibliography(
    joinpath(@__DIR__, "src", "assets", "references.bib"),
    style=:numeric
)

# Build documentation.
@timeit dto "makedocs" makedocs(
    format = Documenter.HTML(
        assets =["assets/citations.css",
        "assets/custom.css"]
    ),
    sitename = "FerriteMultigrid.jl",
    doctest = false,
    warnonly = true,
    draft = liveserver,
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => [
            "tutorials/linear_elasticity.md",
            "tutorials/hyperelasticity.md",
            ],
        "API Reference" => ["api-reference/fe.md",
                             "api-reference/multigrid_problems.md",
                             "api-reference/multilevel.md",
                             "api-reference/pmg_config.md",],
        "references.md",
        ],
    plugins = [
        bibtex_plugin,
    ]
)

# Deploy built documentation (only if not liveserver)
if !liveserver
    @timeit dto "deploydocs" deploydocs(
        repo = "github.com/termi-official/FerriteMultigrid.jl.git",
        devbranch = "main",
        push_preview = true,
        versions = [
            "stable" => "v^",
            "dev" => "dev"
        ]
    )
end


print_timer(dto)
