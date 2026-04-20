"use client";

import type { Components } from "react-markdown";
import ReactMarkdown from "react-markdown";

const components: Components = {
  a({ href, children, ...rest }) {
    const external =
      typeof href === "string" && (href.startsWith("http://") || href.startsWith("https://"));
    return (
      <a
        href={href}
        {...rest}
        {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
      >
        {children}
      </a>
    );
  },
};

export default function AssistantMarkdown({
  content,
  streaming,
}: {
  content: string;
  streaming?: boolean;
}) {
  return (
    <div className="msg-body md-content">
      {content.trim() ? <ReactMarkdown components={components}>{content}</ReactMarkdown> : null}
      {streaming ? <span className="streaming-cursor" aria-hidden /> : null}
    </div>
  );
}
