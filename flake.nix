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
          aiofiles
          aiohttp
          beautifulsoup4
          docstring-parser
          html2text
          humanize
          keyring
          mcp
          oauthlib
          partial-json-parser
          prompt_toolkit
          wcwidth
        ];

        # Extract version from __init__.py
        versionFromInit = builtins.head (
          builtins.match ".*__version__ = \"([^\"]+)\".*"
            (builtins.readFile ./src/tclaude/__init__.py)
        );

        # Your package derivation
        tclaudeProject = pythonPackages.buildPythonPackage rec {
          pname = "tclaude";
          version = versionFromInit;
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
        packages.default = tclaudeProject;

        # Development environment
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pythonPackages.black
            pythonPackages.build
            pythonPackages.flake8
            pythonPackages.mypy
            pythonPackages.pip
            pythonPackages.pytest
            pythonPackages.setuptools
            pythonPackages.wheel
          ] ++ pythonDeps;
        };
      });
}
