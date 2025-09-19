import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/**
 * A utility function to merge Tailwind CSS classes without conflicts.
 * It intelligently combines classes, resolving contradictions.
 * For example, `cn('p-2', 'p-4')` will result in `'p-4'`.
 * @param {...ClassValue[]} inputs - A list of class strings or objects.
 * @returns {string} The merged class string.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
