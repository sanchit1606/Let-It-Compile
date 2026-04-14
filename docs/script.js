/* Navigation and interactivity */

document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.nav-link');
    const docSections = document.querySelectorAll('.doc-section');

    function updateActiveSection(targetId) {
        // Hide all sections
        docSections.forEach(section => {
            section.classList.remove('active');
        });

        // Remove active class from all nav links
        navLinks.forEach(link => {
            link.classList.remove('active');
        });

        // Show target section
        const targetSection = document.getElementById(targetId);
        if (targetSection) {
            targetSection.classList.add('active');
        }

        // Mark nav link as active
        const activeLink = document.querySelector(`a[href="#${targetId}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }

        // Scroll to top of content
        document.querySelector('.content').scrollTop = 0;

        // Save to localStorage for persistence
        localStorage.setItem('lastViewedSection', targetId);
    }

    // Add click listeners to nav links
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            updateActiveSection(targetId);
            window.history.pushState(null, null, `#${targetId}`);
        });
    });

    // Handle hash changes
    window.addEventListener('hashchange', function() {
        const hash = window.location.hash.substring(1);
        if (hash) {
            updateActiveSection(hash);
        }
    });

    // Load last viewed section or default to overview
    const lastViewed = localStorage.getItem('lastViewedSection');
    const initialSection = lastViewed || 'overview';
    
    if (window.location.hash) {
        updateActiveSection(window.location.hash.substring(1));
    } else {
        updateActiveSection(initialSection);
    }

    // Smooth scroll for navigation within content
    document.addEventListener('click', function(e) {
        if (e.target.tagName === 'A' && e.target.href.includes('#')) {
            const href = e.target.getAttribute('href');
            if (href.startsWith('#')) {
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement && targetElement.classList.contains('doc-section')) {
                    e.preventDefault();
                    updateActiveSection(targetId);
                }
            }
        }
    });
});

// Keyboard navigation support
document.addEventListener('keydown', function(e) {
    const navLinks = document.querySelectorAll('.nav-link');
    const currentIndex = Array.from(navLinks).findIndex(link => link.classList.contains('active'));

    if (e.key === 'ArrowDown' && currentIndex < navLinks.length - 1) {
        navLinks[currentIndex + 1].click();
        e.preventDefault();
    } else if (e.key === 'ArrowUp' && currentIndex > 0) {
        navLinks[currentIndex - 1].click();
        e.preventDefault();
    }
});

// Search functionality (optional enhancement)
function searchDocumentation(query) {
    const sections = document.querySelectorAll('.doc-section');
    const query_lower = query.toLowerCase();
    const results = [];

    sections.forEach(section => {
        const text = section.innerText.toLowerCase();
        const title = section.querySelector('h1')?.innerText || '';

        if (text.includes(query_lower)) {
            results.push({
                id: section.id,
                title: title,
                relevance: (text.match(new RegExp(query_lower, 'g')) || []).length
            });
        }
    });

    return results.sort((a, b) => b.relevance - a.relevance);
}

// Table of contents generation (for future enhancement)
function generateTableOfContents() {
    const toc = [];
    const sections = document.querySelectorAll('.doc-section');

    sections.forEach(section => {
        const heading = section.querySelector('h1');
        if (heading) {
            toc.push({
                id: section.id,
                title: heading.innerText,
                level: 1
            });

            const subHeadings = section.querySelectorAll('h2, h3');
            subHeadings.forEach(sub => {
                toc.push({
                    id: sub.innerText.toLowerCase().replace(/\s+/g, '-'),
                    title: sub.innerText,
                    level: sub.tagName === 'H2' ? 2 : 3
                });
            });
        }
    });

    return toc;
}
