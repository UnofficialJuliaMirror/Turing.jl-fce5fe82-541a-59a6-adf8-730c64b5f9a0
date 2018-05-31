using Documenter, Turing

makedocs(
	modules = [Turing],
    format = :html,
    sitename = "Turing.jl",
    pages = [
        "Home" => "index.md",
        "Compiler Notes" => "compiler_notes.md",
        "Semantics Notes" => "semantics_notes.md",
    ],
)
