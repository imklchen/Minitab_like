# Session Management Guide for Minitab-like Application

This guide provides best practices for managing development sessions and preventing data loss when working on the Minitab-like application.

## Preventing and Handling Crashes

### Development Session Management

1. **Break development into chunks**:
   - Use the development_tasks.md file to work on one feature group at a time
   - Complete one logical unit of work before moving to the next
   - Commit changes after completing each logical unit

2. **Implement frequent save points**:
   - The auto-save feature now saves every 5 minutes
   - Manually save important changes with Ctrl+S
   - Use "File > Save As..." to create checkpoints with descriptive names

3. **Run the modularization script**:
   - Use `python modularize_app.py` to split the application into smaller modules
   - This reduces the likelihood of crashes by preventing the IDE from loading the entire codebase at once
   - Work in the modular version when making significant changes

### Resource Management

1. **Memory optimization**:
   - Clear large variables when no longer needed
   - Use generators instead of lists for large data processing
   - Add this line at the end of methods that create plots: `plt.close(fig)`

2. **Limit concurrent applications**:
   - Close memory-intensive applications when working on the application
   - Monitor system resource usage with Task Manager (Windows) or Activity Monitor (Mac)
   - Consider increasing available memory if possible

### Backup and Recovery

1. **Regular backups**:
   - The application now creates auto-save files in the `autosave` directory
   - Create daily backups in a different location
   - Use version control (Git) for more comprehensive backup

2. **Recovery process**:
   - If the application crashes, restart it and check for recovery files
   - If auto-recovery fails, check the `autosave` directory for the most recent auto-save
   - As a last resort, use backups or version control to restore to a previous state

## Development Best Practices

### Code Structure

1. **Modularization**:
   - Keep methods small and focused on a single task
   - Separate UI code from business logic
   - Use clear, consistent naming conventions

2. **Error handling**:
   - Wrap code in try-except blocks to prevent crashes
   - Log errors for later debugging
   - Show user-friendly error messages

3. **Documentation**:
   - Document complex code sections
   - Update documentation when making changes
   - Use clear commit messages when using version control

### Testing

1. **Test early and often**:
   - Test features immediately after implementation
   - Create test cases for edge conditions
   - Verify that error handling works as expected

2. **Incremental testing**:
   - Test one feature at a time
   - Verify that the feature works correctly before moving on
   - Check for unintended side effects

### UI Development

1. **Preview changes**:
   - Use a separate preview window for testing UI changes
   - Keep the main window as stable as possible
   - Implement UI changes incrementally

2. **Performance considerations**:
   - Minimize UI updates during calculations
   - Show progress indicators for long-running operations
   - Consider moving heavy processing to background threads

## Troubleshooting Common Issues

### Application Crashes

1. **Memory-related crashes**:
   - Symptom: Application freezes or crashes when working with large datasets
   - Solution: Implement pagination or data streaming for large data
   - Prevention: Monitor memory usage and optimize resource-intensive operations

2. **UI thread blocking**:
   - Symptom: UI becomes unresponsive during calculations
   - Solution: Move calculations to a background thread
   - Prevention: Use progress dialogs and background processing for long operations

3. **File access issues**:
   - Symptom: Application crashes when trying to access files
   - Solution: Implement robust file access error handling
   - Prevention: Check file permissions and existence before operations

### IDE Crashes

1. **VS Code/Cursor crashes**:
   - Symptom: IDE freezes or crashes when working with large files
   - Solution: Use the modularized version of the application
   - Prevention: Break large files into smaller modules

2. **Extension conflicts**:
   - Symptom: IDE crashes when certain operations are performed
   - Solution: Disable problematic extensions temporarily
   - Prevention: Use minimal extension configurations when working on large projects

## Tools and Resources

1. **Recommended tools**:
   - Memory profilers: `memory_profiler`, `psutil`
   - Performance monitoring: Windows Task Manager, Mac Activity Monitor
   - Code quality: `pylint`, `flake8`

2. **Helpful resources**:
   - PyQt6 documentation: https://doc.qt.io/qtforpython-6/
   - Python memory management: https://docs.python.org/3/c-api/memory.html
   - Matplotlib memory management: https://matplotlib.org/stable/users/faq/howto_faq.html

## Conclusion

Following these session management practices will help prevent data loss and make development more efficient. The modularized structure and auto-save features provide additional protection against crashes, but good development practices remain essential for maintaining a stable application.

If you experience recurring crashes, consider further optimizing memory usage, implementing more background processing, or splitting the application into even smaller components. 