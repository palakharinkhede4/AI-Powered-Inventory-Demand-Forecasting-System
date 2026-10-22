"use client";

import { Sun, Moon, Laptop } from "lucide-react";

export type ThemeMode = "light" | "dark" | "system";

export default function ThemePicker({
  theme,
  onChange,
}: {
  theme: ThemeMode;
  onChange: (theme: ThemeMode) => void;
}) {
  return (
    <div className="theme-picker" role="group" aria-label="Appearance">
      {(
        [
          { value: "light", label: "Light mode", Icon: Sun },
          { value: "dark", label: "Dark mode", Icon: Moon },
          { value: "system", label: "System default", Icon: Laptop },
        ] as const
      ).map(({ value, label, Icon }) => (
        <button
          key={value}
          type="button"
          title={label}
          aria-label={label}
          aria-pressed={theme === value}
          onClick={() => onChange(value)}
        >
          <Icon size={15} />
        </button>
      ))}
    </div>
  );
}
