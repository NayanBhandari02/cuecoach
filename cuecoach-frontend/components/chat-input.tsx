"use client";

import { useState } from "react";

type ChatInputProps = {
  onSend: (question: string) => Promise<void>;
  disabled?: boolean;
};

export default function ChatInput({
  onSend,
  disabled = false,
}: ChatInputProps) {
  const [value, setValue] = useState("");

  async function submitCurrentValue() {
    const question = value.trim();
    if (!question || disabled) return;

    setValue("");
    await onSend(question);
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    await submitCurrentValue();
  }

  async function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      await submitCurrentValue();
    }
  }

  return (
    <form onSubmit={handleSubmit} className="mt-4 flex gap-3">
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question about your billiards documents..."
        rows={3}
        className="flex-1 resize-none rounded-2xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:border-slate-500"
        disabled={disabled}
      />
      <button
        type="submit"
        disabled={disabled}
        className="self-end rounded-2xl bg-blue-600 px-5 py-3 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
      >
        Send
      </button>
    </form>
  );
}