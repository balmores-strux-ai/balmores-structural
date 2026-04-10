import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "BALMORES STRUCTURAL",
    short_name: "Balmores",
    description: "Design brief to PyNite 3D finite element analysis — reactions, drift, member forces.",
    start_url: "/",
    display: "standalone",
    background_color: "#080a0f",
    theme_color: "#080a0f",
  };
}
