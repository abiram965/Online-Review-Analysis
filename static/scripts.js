// Highlight search input when focused
document.addEventListener("DOMContentLoaded", () => {
    const searchInput = document.querySelector('input[type="text"]');
    searchInput.addEventListener("focus", () => {
        searchInput.style.borderColor = "#4CAF50";
    });
    searchInput.addEventListener("blur", () => {
        searchInput.style.borderColor = "#ccc";
    });
});

// Smooth scroll to analysis results
function scrollToResults() {
    const resultsSection = document.getElementById("results");
    if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: "smooth" });
    }
}
