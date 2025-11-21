# Report 1 Conversion Guide

## Converting Report1-submission.md to Word Document

### Step 1: Open the Markdown File
- File location: `Report1-submission.md`
- Contains complete text with figure placeholders

### Step 2: Create Word Document
Options:
1. **Pandoc** (Recommended): 
   ```bash
   pandoc Report1-submission.md -o CSE546_FinalProject_Report1.docx
   ```

2. **Online Converter**: Use https://www.markdowntoword.com/

3. **Manual Copy**: Copy text into Word and format manually

### Step 3: Insert Figures
All figures are in: `results/figures/report1/`

**Insert at these locations:**

1. **Figure 1**: Section 2.2 (after "As shown in Figure 1")
   - File: `figure1_class_distribution.png`
   
2. **Figure 2**: Section 3.2 (after Table 2)
   - File: `figure2_baseline_performance.png`
   
3. **Figure 3**: Section 4.1 (after Methodology paragraph)
   - File: `figure3_normalization_comparison.png`
   
4. **Figure 4**: Section 4.2 (after Variance Analysis)
   - File: `figure4_pca_variance.png`
   
5. **Figure 5**: Section 4.2 (after Figure 4)
   - File: `figure5_pca_performance.png`
   
6. **Figure 6**: Section 4.3 (after Methodology)
   - File: `figure6_feature_selection.png`
   
7. **Figure 7**: Section 5.5 (after heading)
   - File: `figure7_svm_learning_curves.png`
   
8. **Figure 8**: Section 5.4 (after heading)
   - File: `figure8_svm_parameter_impact.png`
   
9. **Figure 9**: Section 5.6 (after heading)
   - File: `figure9_svm_confusion_matrix.png`

### Step 4: Format the Document

**Formatting Guidelines:**
- Font: 12pt (as required)
- Margins: 1 inch all sides
- Line spacing: 1.5 or double
- Tables: Center-align and apply grid borders
- Figures: Center-align with captions below

**Styling:**
- Heading 1: Section titles (e.g., "1. Introduction")
- Heading 2: Subsections (e.g., "2.1 Dataset Overview")
- Heading 3: Sub-subsections (e.g., "2.2.1 Class Details")
- Normal: Body text
- Table style: Professional grid (Light Grid - Accent 1 recommended)

### Step 5: Final Checks

- [ ] All 9 figures inserted and visible
- [ ] All tables formatted properly
- [ ] Page count: 7-10 pages (currently ~8-9 pages with figures)
- [ ] All figure references work (e.g., "As shown in Figure 3...")
- [ ] Font size is 12pt throughout
- [ ] Student name on title page
- [ ] No markdown syntax remaining (e.g., no `**` or `##`)

### Step 6: Save and Submit

- Save as: `CSE546_FinalProject_Report1_Fuentes.docx`
- Alternative format: Can convert to PDF if preferred
- **Do NOT zip** - Submit single file to Blackboard

---

## Quick Word Formatting Tips

### Table Formatting
1. Select table
2. Table Design → Table Styles
3. Choose "Grid Table 4 - Accent 1" or similar
4. Check "Header Row" and "Banded Rows"

### Figure Insertion
1. Place cursor at figure location
2. Insert → Pictures → This Device
3. Select figure file
4. Right-click figure → Size and Position
5. Set width to 6-6.5 inches (maintains readability)
6. Add caption: References → Insert Caption → "Figure X: Description"

### Caption Formatting
- Caption style: Bold
- Position: Below figures
- Numbering: Automatic (Figure 1, Figure 2, etc.)

---

## Expected Final Document Stats

- **Pages**: 8-10 pages (with figures)
- **Word Count**: ~5,500-6,000 words
- **Figures**: 9 (all numbered and captioned)
- **Tables**: 9 (all formatted professionally)
- **Sections**: 8 main sections + appendix
- **References**: 3 academic sources

---

## Troubleshooting

**Problem**: Pandoc not installed
**Solution**: 
```bash
# Windows (PowerShell as Admin)
choco install pandoc

# Or download from: https://pandoc.org/installing.html
```

**Problem**: Figures too large/small
**Solution**: In Word, right-click → Size and Position → Width: 6 inches (maintain aspect ratio)

**Problem**: Tables not aligned
**Solution**: Select table → Layout → AutoFit → AutoFit to Window

**Problem**: Page count too long
**Solution**: Reduce figure sizes slightly or adjust margins to 0.9 inches

---

## Ready to Submit Checklist

- [ ] Document is 7-10 pages
- [ ] All figures inserted and numbered
- [ ] All tables formatted
- [ ] Font is 12pt throughout
- [ ] Name on title page
- [ ] File name follows convention
- [ ] Saved as .docx or .pdf (NOT .zip)
- [ ] Proofread for typos
- [ ] All placeholders replaced with actual content
- [ ] References section complete

---

**Estimated Time**: 30-45 minutes for complete conversion and formatting

