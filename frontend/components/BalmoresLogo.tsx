import Link from "next/link";

type BalmoresLogoProps = {
  /** Primary line — defaults to BALMORES LAB */
  title?: string;
  /** Optional second line under the title */
  subtitle?: string;
  /** nav = compact single-line; header = home page top bar */
  variant?: "nav" | "header";
  href?: string;
  className?: string;
};

function LogoMark({ size = 36 }: { size?: number }) {
  return (
    <svg
      className="balmores-logo__mark"
      width={size}
      height={size}
      viewBox="0 0 48 48"
      aria-hidden
      focusable="false"
    >
      <defs>
        <linearGradient id="bl-gold-inline" x1="8" y1="6" x2="40" y2="42" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#f0d78a" />
          <stop offset="42%" stopColor="#c9a24a" />
          <stop offset="100%" stopColor="#8f6b28" />
        </linearGradient>
        <linearGradient id="bl-shine-inline" x1="14" y1="10" x2="28" y2="24" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#fff" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#fff" stopOpacity="0" />
        </linearGradient>
      </defs>
      <circle cx="24" cy="24" r="23" fill="url(#bl-gold-inline)" />
      <circle cx="24" cy="24" r="23" fill="url(#bl-shine-inline)" />
      <circle cx="24" cy="24" r="21.5" fill="none" stroke="#5c4520" strokeOpacity="0.35" strokeWidth="0.75" />
      <path
        d="M15.5 31.5 L24 13.5 L32.5 31.5"
        fill="none"
        stroke="#141820"
        strokeWidth="2.35"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M17.25 27.75 H30.75"
        fill="none"
        stroke="#141820"
        strokeWidth="2.35"
        strokeLinecap="round"
      />
      <path
        d="M19 24 H29"
        fill="none"
        stroke="#141820"
        strokeWidth="2.35"
        strokeLinecap="round"
      />
      <path
        d="M24 13.5 V31.5"
        fill="none"
        stroke="#141820"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeOpacity="0.55"
      />
    </svg>
  );
}

export default function BalmoresLogo({
  title = "BALMORES LAB",
  subtitle,
  variant = "nav",
  href = "/",
  className = "",
}: BalmoresLogoProps) {
  const markSize = variant === "header" ? 40 : 34;

  return (
    <Link
      href={href}
      className={`balmores-logo balmores-logo--${variant} ${className}`.trim()}
      aria-label="Balmores Lab — home"
    >
      <LogoMark size={markSize} />
      <span className="balmores-logo__text">
        <span className="balmores-logo__title">{title}</span>
        {subtitle ? <span className="balmores-logo__subtitle">{subtitle}</span> : null}
      </span>
    </Link>
  );
}
