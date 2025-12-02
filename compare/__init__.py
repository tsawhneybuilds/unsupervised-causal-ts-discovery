"""
Compare package for multi-algorithm graph comparison using Tetrad's statistics.

This package provides:
- graph_io: Parse Tetrad graph format and convert Python graphs to Tetrad Graph objects
- metrics: Compute comparison statistics (F1Adj, F1Arrow, SHD) via py-tetrad/JPype
- algorithms: Wrappers for various causal discovery algorithms
- runner: Orchestrate algorithm runs and output comparison tables
"""

import os
import sys
import subprocess
import platform

# =============================================================================
# Tigramite path setup
# =============================================================================
# Add the Causal_with_Tigramite folder to sys.path so tigramite can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tigramite_path = os.path.join(_project_root, 'Causal_with_Tigramite')
if _tigramite_path not in sys.path and os.path.exists(_tigramite_path):
    sys.path.insert(0, _tigramite_path)

# JPype and Tetrad initialization
_jvm_started = False


def find_java_home():
    """
    Find JAVA_HOME using multiple methods.
    
    Returns:
        str: Path to Java home directory, or None if not found
    """
    # 1. Check JAVA_HOME environment variable first
    java_home = os.environ.get('JAVA_HOME')
    if java_home and os.path.exists(java_home):
        return java_home
    
    # 2. On ARM macOS, check known ARM Java installations FIRST
    #    This prevents picking up x86_64 Java from Homebrew Intel
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # Known ARM Java 21 installation paths (most common first)
        arm_java_paths = [
            # Eclipse Temurin (ARM)
            '/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home',
            # Amazon Corretto (ARM)
            '/Library/Java/JavaVirtualMachines/amazon-corretto-21.jdk/Contents/Home',
            # Zulu (ARM)
            '/Library/Java/JavaVirtualMachines/zulu-21.jdk/Contents/Home',
            # Oracle GraalVM (ARM)
            '/Library/Java/JavaVirtualMachines/graalvm-21/Contents/Home',
            # Homebrew ARM (Apple Silicon path)
            '/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home',
            '/opt/homebrew/Cellar/openjdk@21/21.0.2/libexec/openjdk.jdk/Contents/Home',
        ]
        
        for path in arm_java_paths:
            if os.path.exists(path):
                return path
        
        # Also check /Library/Java/JavaVirtualMachines/ for any Java 21
        jvm_dir = '/Library/Java/JavaVirtualMachines'
        if os.path.exists(jvm_dir):
            for name in os.listdir(jvm_dir):
                if '21' in name:
                    candidate = os.path.join(jvm_dir, name, 'Contents', 'Home')
                    if os.path.exists(candidate):
                        return candidate
    
    # 3. On macOS, use /usr/libexec/java_home (but skip x86_64 on ARM)
    if platform.system() == 'Darwin':
        is_arm = platform.machine() == 'arm64'
        
        for version in ['21', '17']:
            try:
                result = subprocess.run(
                    ['/usr/libexec/java_home', '-v', version],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    java_home = result.stdout.strip()
                    if java_home and os.path.exists(java_home):
                        # Skip x86_64 Java on ARM Mac
                        if is_arm and '/usr/local/Cellar/' in java_home:
                            continue
                        return java_home
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                continue
        
        # Fallback: try default java_home (any version)
        try:
            result = subprocess.run(
                ['/usr/libexec/java_home'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                java_home = result.stdout.strip()
                if java_home and os.path.exists(java_home):
                    # Still skip x86_64 on ARM
                    if is_arm and '/usr/local/Cellar/' in java_home:
                        pass  # Don't return this
                    else:
                        return java_home
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
    
    # 3. On Linux, check common paths
    if platform.system() == 'Linux':
        common_paths = [
            '/usr/lib/jvm/java-21-openjdk-amd64',
            '/usr/lib/jvm/java-17-openjdk-amd64',
            '/usr/lib/jvm/java-11-openjdk-amd64',
            '/usr/lib/jvm/default-java',
            '/usr/lib/jvm/java-openjdk',
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
    
    # 4. On Windows, check registry or common paths
    if platform.system() == 'Windows':
        common_paths = [
            os.path.join(os.environ.get('ProgramFiles', ''), 'Java'),
            os.path.join(os.environ.get('ProgramFiles(x86)', ''), 'Java'),
        ]
        for base_path in common_paths:
            if os.path.exists(base_path):
                # Find the newest Java version
                try:
                    versions = os.listdir(base_path)
                    for v in sorted(versions, reverse=True):
                        java_path = os.path.join(base_path, v)
                        if os.path.exists(os.path.join(java_path, 'bin', 'java.exe')):
                            return java_path
                except Exception:
                    continue
    
    # 5. Try 'which java' and derive JAVA_HOME
    try:
        result = subprocess.run(
            ['which', 'java'] if platform.system() != 'Windows' else ['where', 'java'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            java_bin = result.stdout.strip().split('\n')[0]
            # Follow symlinks to find real path
            java_bin = os.path.realpath(java_bin)
            # Go up from bin/java to JAVA_HOME
            if 'bin' in java_bin:
                java_home = os.path.dirname(os.path.dirname(java_bin))
                if os.path.exists(java_home):
                    return java_home
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return None


def get_jvm_path(java_home):
    """
    Get the JVM library path from JAVA_HOME.
    
    Args:
        java_home: Path to Java home directory
        
    Returns:
        str: Path to JVM library, or None if not found
    """
    system = platform.system()
    
    if system == 'Darwin':
        # macOS paths - order matters, try most common first
        # Note: Homebrew OpenJDK puts libs in Contents/Home/lib/
        possible_paths = [
            os.path.join(java_home, 'lib', 'libjli.dylib'),
            os.path.join(java_home, 'lib', 'server', 'libjvm.dylib'),
            # Homebrew OpenJDK structure
            os.path.join(java_home, 'lib', 'jli', 'libjli.dylib'),
            # JDK bundle structure
            os.path.join(java_home, 'Contents', 'Home', 'lib', 'libjli.dylib'),
            os.path.join(java_home, 'Contents', 'Home', 'lib', 'server', 'libjvm.dylib'),
            # Legacy paths
            os.path.join(java_home, 'jre', 'lib', 'libjli.dylib'),
            os.path.join(java_home, 'jre', 'lib', 'server', 'libjvm.dylib'),
        ]
    elif system == 'Linux':
        # Linux paths
        possible_paths = [
            os.path.join(java_home, 'lib', 'server', 'libjvm.so'),
            os.path.join(java_home, 'lib', 'amd64', 'server', 'libjvm.so'),
            os.path.join(java_home, 'jre', 'lib', 'amd64', 'server', 'libjvm.so'),
        ]
    elif system == 'Windows':
        # Windows paths
        possible_paths = [
            os.path.join(java_home, 'bin', 'server', 'jvm.dll'),
            os.path.join(java_home, 'jre', 'bin', 'server', 'jvm.dll'),
        ]
    else:
        return None
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def init_jvm():
    """
    Initialize the JVM with Tetrad jar for py-tetrad.
    Must be called before using any Tetrad functionality.
    
    Raises:
        RuntimeError: If JVM cannot be initialized (Java not found)
    """
    global _jvm_started
    if _jvm_started:
        return
    
    import jpype
    
    if jpype.isJVMStarted():
        _jvm_started = True
        return
    
    # Find Java
    java_home = find_java_home()
    
    is_arm_mac = platform.system() == 'Darwin' and platform.machine() == 'arm64'
    
    if not java_home:
        separator = "=" * 70
        if is_arm_mac:
            error_msg = f"""
{separator}
ERROR: Java 21+ (ARM) not found!
{separator}

The comparison framework requires Java 21+ for Tetrad statistics.
You're on an ARM Mac (M1/M2/M3) and need ARM-native Java.

To fix this, install ARM Java 21:

  Option 1 - Amazon Corretto (recommended):
    brew install --cask corretto@21

  Option 2 - Eclipse Temurin:
    brew install --cask temurin@21

  Option 3 - Download directly:
    https://adoptium.net/temurin/releases/?arch=aarch64&os=mac

Then add to ~/.zshrc:
    export JAVA_HOME=$(/usr/libexec/java_home -v 21)

Note: x86_64 Java (via Rosetta) won't work properly.
{separator}"""
        else:
            error_msg = f"""
{separator}
ERROR: Java 21+ not found!
{separator}

The comparison framework requires Java 21+ for Tetrad statistics.

To fix this, install Java and set JAVA_HOME:

  macOS (with Homebrew):
    brew install openjdk@21
    export JAVA_HOME=$(/usr/libexec/java_home -v 21)

  Linux (Ubuntu/Debian):
    sudo apt install openjdk-21-jdk
    export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

  Windows:
    Download from https://adoptium.net/
    Set JAVA_HOME in System Environment Variables

Add the export command to your shell profile (~/.zshrc or ~/.bashrc)
for a permanent fix.
{separator}"""
        raise RuntimeError(error_msg)
    
    # Set JAVA_HOME if not already set
    if 'JAVA_HOME' not in os.environ:
        os.environ['JAVA_HOME'] = java_home
        print(f"Setting JAVA_HOME to: {java_home}")
    
    # Find JVM path
    jvm_path = get_jvm_path(java_home)
    
    # Find tetrad jar from py-tetrad installation
    try:
        import pytetrad
        pytetrad_dir = os.path.dirname(pytetrad.__file__)
        
        # Check possible jar locations
        jar_locations = [
            os.path.join(pytetrad_dir, "resources", "tetrad-current.jar"),
            os.path.join(pytetrad_dir, "tetrad-current.jar"),
            os.path.join(pytetrad_dir, "resources", "tetrad-lib-current.jar"),
        ]
        
        tetrad_jar = None
        for jar_path in jar_locations:
            if os.path.exists(jar_path):
                tetrad_jar = jar_path
                break
        
        if tetrad_jar:
            # Try to start JVM - first with explicit path, then without
            try:
                if jvm_path:
                    jpype.startJVM(jvm_path, classpath=[tetrad_jar])
                else:
                    jpype.startJVM(classpath=[tetrad_jar])
            except (FileNotFoundError, OSError) as e:
                # If explicit path fails (e.g., architecture mismatch), 
                # let JPype find JVM automatically
                print(f"Note: Explicit JVM path failed ({e}), trying automatic detection...")
                if not jpype.isJVMStarted():
                    jpype.startJVM(classpath=[tetrad_jar])
        else:
            # Let py-tetrad handle JVM startup
            from pytetrad.tools import TetradSearch
            import pandas as pd
            dummy_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            _ = TetradSearch(dummy_df)
            
    except FileNotFoundError as e:
        # Check if this is an ARM Mac with x86_64 Java
        is_arm_mac = platform.system() == 'Darwin' and platform.machine() == 'arm64'
        is_intel_java = java_home and '/usr/local/Cellar/' in java_home
        
        separator = "=" * 70
        if is_arm_mac and is_intel_java:
            error_msg = f"""
{separator}
ERROR: Architecture mismatch - x86_64 Java on ARM Mac!
{separator}

Detected Java: {java_home}
This is x86_64 Java which won't work on your ARM Mac (M1/M2/M3).

You need to install ARM-native Java 21:

  Option 1 - Amazon Corretto (recommended):
    brew install --cask corretto@21

  Option 2 - Eclipse Temurin:
    brew install --cask temurin@21

Then add to ~/.zshrc:
    export JAVA_HOME=$(/usr/libexec/java_home -v 21)

And restart your terminal.
{separator}"""
        else:
            error_msg = f"""
{separator}
ERROR: JVM library not found!
{separator}

Java home detected: {java_home}
But the JVM library could not be loaded.

This usually means Java is not properly installed.

Try reinstalling Java:
  macOS: brew reinstall openjdk@21
  Then: export JAVA_HOME=$(/usr/libexec/java_home -v 21)

Original error: {e}
{separator}"""
        raise RuntimeError(error_msg) from e
    except Exception as e:
        separator = "=" * 70
        error_msg = f"""
{separator}
ERROR: Failed to initialize JVM!
{separator}

Java home: {java_home}
Error: {e}

Please ensure:
1. Java 21+ is properly installed (ARM version for M1/M2/M3 Macs)
2. JAVA_HOME is set correctly
3. py-tetrad is installed: pip install py-tetrad
{separator}"""
        raise RuntimeError(error_msg) from e
    
    _jvm_started = True
    print(f"JVM initialized successfully (Java: {java_home})")


def is_jvm_started():
    """Check if JVM has been started."""
    import jpype
    return jpype.isJVMStarted()


__all__ = ['init_jvm', 'is_jvm_started', 'find_java_home']
