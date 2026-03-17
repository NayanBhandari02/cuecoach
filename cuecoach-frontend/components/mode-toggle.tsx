"use client";

import type { ChatMode } from "@/types/chat";

type ModeToggleProps = {
  value: ChatMode;
  onChange: (mode: ChatMode) => void;
};

export default function ModeToggle({
  value,
  onChange,
}: ModeToggleProps) {
  return (
    <div className="inline-flex rounded-xl border border-slate-700 bg-slate-950 p-1">
      {(["strict", "explain"] as ChatMode[]).map((mode) => {
        const active = value === mode;

        return (
          <button
            key={mode}
            type="button"
            onClick={() => onChange(mode)}
            className={[
              "rounded-lg px-4 py-2 text-sm font-medium transition",
              active
                ? "bg-blue-600 text-white"
                : "bg-transparent text-slate-300 hover:bg-slate-800",
            ].join(" ")}
          >
            {mode === "strict" ? "Strict" : "Explain"}
          </button>
        );
      })}
    </div>
  );
}