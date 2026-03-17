"use client";

import type { ChatMessage } from "@/types/chat";
import { useEffect, useRef } from "react";

type ChatWindowProps = {
  messages: ChatMessage[];
  isLoading: boolean;
};

export default function ChatWindow({
  messages,
  isLoading,
}: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex h-[60vh] flex-col gap-4 overflow-y-auto rounded-2xl border border-slate-800 bg-slate-950 p-4 shadow-sm">
      {messages.length === 0 ? (
        <div className="m-auto max-w-xl text-center text-slate-400">
          <p className="text-lg font-medium text-slate-100">Start the chat</p>
          <p className="mt-2 text-sm">
            Ask a question. For example: How to play a rail first shot?
          </p>
        </div>
      ) : (
        messages.map((message, index) => {
          const isUser = message.role === "user";

          return (
            <div
              key={`${message.role}-${index}`}
              className={`flex ${isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={[
                  "max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-6 whitespace-pre-wrap",
                  isUser
                    ? "bg-blue-600 text-white"
                    : "bg-slate-800 text-slate-100",
                ].join(" ")}
              >
                {message.content}
              </div>
            </div>
          );
        })
      )}

      {isLoading && (
        <div className="flex justify-start">
          <div className="max-w-[85%] rounded-2xl bg-slate-800 px-4 py-3 text-sm text-slate-300">
            Thinking...
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}