/**
 * FL HeartMuLa Conditioning Node - Custom Frontend Extension
 * Adds a popup lyrics editor with section buttons for easy song structure building.
 *
 * Uses a modal popup approach for compatibility with both Legacy and V2 modes.
 */

import { app } from "/scripts/app.js";

// Section configuration for song structure
// These are the officially supported markers from HeartMuLa documentation
const SONG_SECTIONS = [
    { label: "[Intro]", color: "#6b5b95" },
    { label: "[Verse]", color: "#88b04b" },
    { label: "[Prechorus]", color: "#f7cac9" },
    { label: "[Chorus]", color: "#92a8d1" },
    { label: "[Bridge]", color: "#955251" },
    { label: "[Outro]", color: "#b565a7" },
    { label: "[Instrumental]", color: "#45b8ac" },
];

/**
 * Lyrics Editor Modal Class
 */
class LyricsEditorModal {
    constructor(node, lyricsWidget) {
        this.node = node;
        this.lyricsWidget = lyricsWidget;
        this.createModal();
        this.setupKeyboardHandlers();
    }

    createModal() {
        // Create overlay
        this.overlay = document.createElement("div");
        this.overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(4px);
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: hmFadeIn 0.2s ease-out;
        `;

        // Create container
        this.container = document.createElement("div");
        this.container.style.cssText = `
            background: linear-gradient(145deg, #2d2d2d, #252525);
            border-radius: 12px;
            border: 1px solid #3a3a3a;
            width: 1000px;
            height: 80%;
            max-width: 95%;
            max-height: 90%;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8);
            animation: hmSlideIn 0.3s ease-out;
        `;

        // Create header
        this.createHeader();

        // Create section buttons
        this.createSectionButtons();

        // Create textarea
        this.createTextarea();

        // Create footer
        this.createFooter();

        this.overlay.appendChild(this.container);

        // Close on overlay click
        this.overlay.addEventListener("click", (e) => {
            if (e.target === this.overlay) {
                this.close();
            }
        });
    }

    createHeader() {
        const header = document.createElement("div");
        header.style.cssText = `
            padding: 16px 20px;
            border-bottom: 1px solid #404040;
            display: flex;
            justify-content: space-between;
            align-items: center;
        `;

        const title = document.createElement("h2");
        title.textContent = "Lyrics Editor";
        title.style.cssText = `
            margin: 0;
            color: #fff;
            font-size: 18px;
            font-weight: 600;
        `;

        const subtitle = document.createElement("div");
        subtitle.textContent = "Click section buttons to insert markers • Auto-saves on close • ESC to cancel without saving";
        subtitle.style.cssText = `
            color: #888;
            font-size: 12px;
            margin-top: 4px;
        `;

        const titleContainer = document.createElement("div");
        titleContainer.appendChild(title);
        titleContainer.appendChild(subtitle);

        const closeBtn = document.createElement("button");
        closeBtn.innerHTML = "✕";
        closeBtn.style.cssText = `
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            width: 32px;
            height: 32px;
            font-size: 16px;
            transition: all 0.2s ease;
        `;
        closeBtn.onmouseover = () => {
            closeBtn.style.background = "rgba(255, 77, 77, 0.8)";
        };
        closeBtn.onmouseout = () => {
            closeBtn.style.background = "rgba(255, 255, 255, 0.05)";
        };
        closeBtn.onclick = () => this.close();

        header.appendChild(titleContainer);
        header.appendChild(closeBtn);
        this.container.appendChild(header);
    }

    createSectionButtons() {
        const toolbar = document.createElement("div");
        toolbar.style.cssText = `
            padding: 12px 20px;
            background: rgba(0, 0, 0, 0.2);
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            border-bottom: 1px solid #333;
        `;

        const label = document.createElement("div");
        label.textContent = "Insert Section:";
        label.style.cssText = `
            width: 100%;
            font-size: 11px;
            color: #888;
            margin-bottom: 4px;
        `;
        toolbar.appendChild(label);

        SONG_SECTIONS.forEach(({ label: sectionLabel, color }) => {
            const btn = document.createElement("button");
            btn.textContent = sectionLabel;
            btn.style.cssText = `
                padding: 6px 12px;
                background: ${color}22;
                color: #ddd;
                border: 1px solid ${color};
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                font-family: monospace;
                transition: all 0.15s ease;
            `;

            btn.onmouseenter = () => {
                btn.style.background = `${color}55`;
                btn.style.transform = "translateY(-1px)";
            };
            btn.onmouseleave = () => {
                btn.style.background = `${color}22`;
                btn.style.transform = "translateY(0)";
            };

            btn.onclick = () => {
                this.insertSection(sectionLabel);
            };

            toolbar.appendChild(btn);
        });

        this.container.appendChild(toolbar);
    }

    createTextarea() {
        const textareaContainer = document.createElement("div");
        textareaContainer.style.cssText = `
            flex: 1;
            padding: 16px 20px;
            min-height: 400px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        `;

        this.textarea = document.createElement("textarea");
        this.textarea.value = this.lyricsWidget.value || "";
        this.textarea.placeholder = "[Verse]\nWrite your lyrics here...\n\n[Chorus]\nAdd your chorus...";
        this.textarea.style.cssText = `
            flex: 1;
            width: 100%;
            height: 100%;
            padding: 16px;
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 6px;
            color: #eee;
            font-family: monospace;
            font-size: 15px;
            line-height: 1.6;
            resize: none;
            outline: none;
            overflow-y: auto;
        `;

        this.textarea.onfocus = () => {
            this.textarea.style.borderColor = "#666";
        };
        this.textarea.onblur = () => {
            this.textarea.style.borderColor = "#444";
        };

        textareaContainer.appendChild(this.textarea);
        this.container.appendChild(textareaContainer);
    }

    createFooter() {
        const footer = document.createElement("div");
        footer.style.cssText = `
            padding: 16px 20px;
            border-top: 1px solid #404040;
            display: flex;
            justify-content: flex-end;
            gap: 12px;
        `;

        const cancelBtn = document.createElement("button");
        cancelBtn.textContent = "Cancel";
        cancelBtn.style.cssText = `
            padding: 8px 20px;
            background: #444;
            border: none;
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        `;
        cancelBtn.onmouseover = () => {
            cancelBtn.style.background = "#555";
        };
        cancelBtn.onmouseout = () => {
            cancelBtn.style.background = "#444";
        };
        cancelBtn.onclick = () => this.close(false);  // Don't save on cancel

        const saveBtn = document.createElement("button");
        saveBtn.textContent = "Save Lyrics";
        saveBtn.style.cssText = `
            padding: 8px 20px;
            background: #4ECDC4;
            border: none;
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        `;
        saveBtn.onmouseover = () => {
            saveBtn.style.background = "#5fd9d0";
        };
        saveBtn.onmouseout = () => {
            saveBtn.style.background = "#4ECDC4";
        };
        saveBtn.onclick = () => this.close(true);  // Save and close

        footer.appendChild(cancelBtn);
        footer.appendChild(saveBtn);
        this.container.appendChild(footer);
    }

    insertSection(sectionLabel) {
        const start = this.textarea.selectionStart;
        const end = this.textarea.selectionEnd;
        const text = this.textarea.value;

        // Build insertion with smart newline handling
        let insertion = sectionLabel + "\n";

        // Add leading newline if not at start and previous char isn't newline
        if (start > 0 && text[start - 1] !== "\n") {
            insertion = "\n" + insertion;
        }

        // Add extra newline for visual separation
        if (start > 1 && text[start - 1] !== "\n") {
            insertion = "\n" + insertion;
        }

        this.textarea.value = text.slice(0, start) + insertion + text.slice(end);

        // Move cursor after the insertion
        const newPos = start + insertion.length;
        this.textarea.selectionStart = newPos;
        this.textarea.selectionEnd = newPos;
        this.textarea.focus();
    }

    save() {
        // Update the widget value
        this.lyricsWidget.value = this.textarea.value;

        // Trigger the widget's callback if it exists
        if (this.lyricsWidget.callback) {
            this.lyricsWidget.callback(this.textarea.value);
        }

        // Also update inputEl if it exists (for the actual textarea in the node)
        if (this.lyricsWidget.inputEl) {
            this.lyricsWidget.inputEl.value = this.textarea.value;
            this.lyricsWidget.inputEl.dispatchEvent(new Event("input", { bubbles: true }));
        }
    }

    setupKeyboardHandlers() {
        this.keydownHandler = (e) => {
            // Escape to close without saving
            if (e.key === "Escape") {
                this.close(false);
            }
            // Ctrl/Cmd + S to save and close
            if ((e.ctrlKey || e.metaKey) && e.key === "s") {
                e.preventDefault();
                this.close(true);
            }
        };

        document.addEventListener("keydown", this.keydownHandler);
    }

    show() {
        // Add CSS animations if not already added
        if (!document.getElementById("heartmula-modal-styles")) {
            const style = document.createElement("style");
            style.id = "heartmula-modal-styles";
            style.textContent = `
                @keyframes hmFadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                @keyframes hmSlideIn {
                    from {
                        opacity: 0;
                        transform: scale(0.95) translateY(-20px);
                    }
                    to {
                        opacity: 1;
                        transform: scale(1) translateY(0);
                    }
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(this.overlay);

        // Focus the textarea
        setTimeout(() => {
            this.textarea.focus();
            // Move cursor to end
            this.textarea.selectionStart = this.textarea.value.length;
            this.textarea.selectionEnd = this.textarea.value.length;
        }, 100);
    }

    close(shouldSave = true) {
        // Auto-save unless explicitly cancelled
        if (shouldSave) {
            this.save();
        }

        // Remove keyboard handler
        document.removeEventListener("keydown", this.keydownHandler);

        // Fade out animation
        this.overlay.style.animation = "hmFadeIn 0.15s ease-in reverse";
        this.container.style.animation = "hmSlideIn 0.15s ease-in reverse";

        setTimeout(() => {
            if (this.overlay.parentNode) {
                document.body.removeChild(this.overlay);
            }
        }, 150);
    }
}

// Register the extension
app.registerExtension({
    name: "FL.HeartMuLa.Conditioning",

    async nodeCreated(node) {
        if (node.comfyClass !== "FL_HeartMuLa_Conditioning") {
            return;
        }

        // Find the lyrics widget
        const lyricsWidget = node.widgets?.find(w => w.name === "lyrics");
        if (!lyricsWidget) {
            console.warn("[FL HeartMuLa] Could not find lyrics widget");
            return;
        }

        // Add "Edit Lyrics" button
        node.addWidget("button", "Edit Lyrics", null, () => {
            const modal = new LyricsEditorModal(node, lyricsWidget);
            modal.show();
        });

        console.log("[FL HeartMuLa] Lyrics editor button added");
    },

    async setup() {
        console.log("[FL HeartMuLa] Frontend extension loaded");
    }
});
