{
  description = "Flakebox Project template";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    flakebox.url = "github:rustshop/flakebox";
    flakebox.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      flakebox,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        projectName = "whisper-real-time";
        pkgs = nixpkgs.legacyPackages.${system};

        flakeboxLib = flakebox.lib.${system} {
          config = {
            github.ci.enable = true;
            github.ci.workflows.flakebox-flakehub-publish.enable = false;
            semgrep.enable = false;
          };
        };

        buildPaths = [
          "Cargo.toml"
          "Cargo.lock"
          "src"
        ];

        buildSrc = flakeboxLib.filterSubPaths {
          root = builtins.path {
            name = projectName;
            path = ./.;
          };
          paths = buildPaths;
        };

        multiBuild = (flakeboxLib.craneMultiBuild { }) (
          craneLib':
          let
            craneLib = (
              craneLib'.overrideArgs {
                pname = projectName;
                src = buildSrc;
                nativeBuildInputs = [
                  pkgs.cmake
                  pkgs.pkg-config
                  pkgs.shaderc
                ];
                buildInputs = [
                  pkgs.alsa-lib
                  pkgs.vulkan-headers
                  pkgs.vulkan-loader
                ];
              }
            );
          in
          {
            ${projectName} = craneLib.buildPackage { };
          }
        );
      in
      {
        packages.default = multiBuild.${projectName};

        legacyPackages = multiBuild;

        devShells = flakeboxLib.mkShells {
          inputsFrom = [ multiBuild.${projectName} ];
          packages = [ pkgs.ffmpeg ];
        };
      }
    );
}
