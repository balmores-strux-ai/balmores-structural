import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Balmores Lab - Louie Doniego Balmores",
    short_name: "Balmores Lab",
    description:
      "Official site of Louie Doniego Balmores - Structural Engineer & AI Researcher. AI-driven structural optimization and PyNite 3D finite element analysis.",
    start_url: "/",
    display: "standalone",
    background_color: "#080a0f",
    theme_color: "#080a0f",
    categories: ["engineering", "productivity", "science"],
  };
}
