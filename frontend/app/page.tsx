import type { Metadata } from "next";
import HomePageClient from "@/components/HomePageClient";
import {
  DEFAULT_DESCRIPTION,
  JOB_TITLE,
  PERSON_NAME,
  SITE_NAME,
  SITE_URL,
} from "@/lib/seo";

export const metadata: Metadata = {
  title: `${PERSON_NAME} - ${JOB_TITLE}`,
  description: DEFAULT_DESCRIPTION,
  alternates: { canonical: "/" },
  openGraph: {
    url: SITE_URL,
    title: `${PERSON_NAME} - ${JOB_TITLE} | ${SITE_NAME}`,
    description: DEFAULT_DESCRIPTION,
  },
  twitter: {
    title: `${PERSON_NAME} - ${JOB_TITLE}`,
    description: DEFAULT_DESCRIPTION,
  },
};

export default function HomePage() {
  return <HomePageClient />;
}
