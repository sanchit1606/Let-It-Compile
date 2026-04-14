# Let It Compile: An RL Approach to Adaptive Register Allocation for GPU Kernel Optimization Across the Stack - Documentation

This folder contains the complete static documentation for the project.

## Quick Start

### Opening the Documentation

Simply open `index.html` in your web browser:

- **Windows:** Double-click `index.html`
- **macOS/Linux:** Open terminal and run: `open index.html` (macOS) or `xdg-open index.html` (Linux)
- **Any OS:** Right-click → "Open with" → Select your browser

Or serve it locally with Python:

```bash
# Python 3.10+
cd docs
python -m http.server 8000
# Visit http://localhost:8000 in your browser
```

## Documentation Structure

The documentation is organized into logical sections accessible from the left sidebar:

### Getting Started
- **Overview** - Project goals and technology stack
- **Quick Start** - Common commands and next steps
- **Installation** - Setup and prerequisites

### Core Concepts
- **Registers & Occupancy** - GPU memory hierarchy and occupancy explained
- **Compiler Knobs** - How we tune kernel compilation
- **GPU Metrics** - Hardware counters and profiling

### Phases (Complete Project Pipeline)
- **Phase 0** - Baseline measurement sweep
- **Phase 1** - Hardware counter collection
- **Phase 2** - Kernel correctness validation
- **Phase 3** - RL environment and PPO training

### Advanced Topics
- **Kernel Implementations** - Details on GEMM, Reduction, Softmax
- **Profiling & Metrics** - Timing, CUPTI, NVML explained
- **Configuration** - Customization options

### Artifacts & Results
- **CSV Schemas** - Exact column meanings for output files
- **Interpreting Results** - How to analyze experiment results

### Reference
- **API Reference** - Code examples and function usage
- **Troubleshooting** - Common issues and solutions
- **Developer Info** - Project structure and contribution guidelines

## Features

- **Dark Theme (NVIDIA Style):** Professional dark UI with green accents
- **Left Sidebar Navigation:** Quick access to all sections
- **Responsive Design:** Works on desktop, tablet, and mobile
- **Keyboard Navigation:** Use arrow keys to navigate between sections
- **Persistent State:** Your last viewed section is remembered
- **Code Syntax Highlighting:** Clear formatting for commands and code
- **Tables & Diagrams:** Organized info on CSV schemas and architectures

## Browser Compatibility

- Chrome/Chromium (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Future Enhancements

As mentioned in the documentation, the following can be added:

1. **Architecture Diagrams** - System architecture, data flow, and phase pipeline (Mermaid/SVG)
2. **Performance Graphs** - Visualizations of Phase 0/1/3 results
3. **Search Functionality** - Full-text search across all sections
4. **Dark Mode Toggle** - Light/dark theme switching
5. **API Interactive Console** - Try API calls interactively

## File Structure

```
docs/
├── index.html          # Main documentation page
├── styles.css          # NVIDIA-themed styling
├── script.js           # Navigation and interactivity
└── README.md           # This file
```

## Styling Notes

The documentation uses NVIDIA's official color palette:
- **Primary Green:** `#76b900` (used for highlights, headings, borders)
- **Light Green:** `#96d700` (used for links, hover states)
- **Dark Background:** `#1a1a1a` (main background)
- **Text Color:** `#e0e0e0` (primary text)

## Mobile Usage

On smaller screens:
- Sidebar becomes collapsible
- Layout adapts to single-column view
- Touch-friendly navigation
- Readable font sizes maintained

## Adding New Sections

To add new documentation sections:

1. Add a new `<section id="unique-id" class="doc-section">` in `index.html`
2. Add corresponding `<li><a href="#unique-id" class="nav-link">...</a></li>` in the sidebar nav
3. Style follows existing patterns automatically
4. Navigation and keyboard shortcuts work immediately

## Accessibility

- Proper semantic HTML structure
- Color contrast meets WCAG standards
- Keyboard navigation support (arrow keys)
- Screen reader friendly
- No JavaScript required for basic reading (graceful degradation)

## Offline Usage

This documentation is completely static and works offline:
- No network requests required
- All resources included locally
- Perfect for archiving or sharing

## Contact & Support

For documentation improvements or feedback, see the Developer Info section in the documentation itself.

---

**Last Updated:** April 2026  
**Project:** Let It Compile: An RL Approach to Adaptive Register Allocation for GPU Kernel Optimization Across the Stack  
**Author:** Sanchit Nipanikar
