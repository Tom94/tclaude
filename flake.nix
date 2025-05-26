{
  description = "Python project with dev and install environments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python313;
        pythonPackages = python.pkgs;

        # Define your Python dependencies here
        pythonDeps = with pythonPackages; [
          aiohttp
          beautifulsoup4
          html2text
          partial-json-parser
          prompt_toolkit
          wcwidth
        ];

        # Your package derivation
        taiProject = pythonPackages.buildPythonPackage rec {
          pname = "tai";
          version = "0.1.0";
          format = "pyproject";
          src = ./.;

          propagatedBuildInputs = pythonDeps;

          # Optional: specify build dependencies
          nativeBuildInputs = with pythonPackages; [
            setuptools
            wheel
          ];

          # checkInputs = with pythonPackages; [
          #   pytest
          # ];

          # doCheck = true;
          # checkPhase = ''
          #   pytest
          # '';
        };

      in
      {
        # System installation
        packages.default = taiProject;

        # Development environment
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pythonPackages.pip
            pythonPackages.setuptools
            pythonPackages.wheel
            pythonPackages.pytest
            pythonPackages.black
            pythonPackages.flake8
            pythonPackages.mypy
          ] ++ pythonDeps;

          shellHook = ''
            echo "Python development environment"
            echo "Python version: $(python --version)"

            # Create virtual environment if it doesn't exist
            if [ ! -d .venv ]; then
              python -m venv .venv
            fi

            source .venv/bin/activate
            pip install -e .
          '';
        };
      });
}
