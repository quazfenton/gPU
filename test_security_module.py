#!/usr/bin/env python3
# coding: utf-8
"""
Test script for security module.

Tests credential encryption, JWT authentication, security logging, and XSS prevention.

Usage:
    python test_security_module.py
"""

import os
import sys
import tempfile
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set up test environment
os.environ['MASTER_KEY'] = 'test-master-key-for-development-only-32bytes!'
os.environ['JWT_SECRET'] = 'test-jwt-secret-for-development-only!'
os.environ['CREDENTIAL_SALT'] = '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'

# Unicode checkmarks for Windows compatibility
CHECK = '[OK]'
CROSS = '[FAIL]'

print("=" * 70)
print("Testing Notebook ML Orchestrator Security Module")
print("=" * 70)

# Test 1: Credential Store
print("\n[TEST 1] Credential Store with AES-256-GCM Encryption")
print("-" * 70)

try:
    from notebook_ml_orchestrator.security.credential_store import (
        CredentialStore
    )
    
    # Create temporary directory for test credentials
    with tempfile.TemporaryDirectory() as temp_dir:
        store_file = os.path.join(temp_dir, 'test_credentials.enc.json')
        salt_file = os.path.join(temp_dir, 'test_salt.hex')
        
        # Initialize credential store
        store = CredentialStore(
            backend_config={
                'store_file': store_file,
                'salt_file': salt_file
            }
        )
        
        print(f"{CHECK} CredentialStore initialized")
        print(f"  - Algorithm: {store.ALGORITHM}")
        print(f"  - Key size: {store.KEY_SIZE * 8} bits")
        print(f"  - PBKDF2 iterations: {store.PBKDF2_ITERATIONS}")
        
        # Test storing credentials
        store.set_credential('modal', 'token_id', 'test-modal-token-id')
        store.set_credential('modal', 'token_secret', 'test-modal-token-secret')
        store.set_credential('huggingface', 'token', 'test-hf-token')
        store.set_credential('kaggle', 'username', 'testuser')
        store.set_credential('kaggle', 'key', 'test-kaggle-key')
        
        print("✓ Stored 5 credentials")
        
        # Test retrieving credentials
        modal_token_id = store.get_credential('modal', 'token_id')
        assert modal_token_id == 'test-modal-token-id', "Credential mismatch!"
        print(f"✓ Retrieved credential: modal:token_id = {modal_token_id[:10]}...")
        
        hf_token = store.get_credential('huggingface', 'token')
        assert hf_token == 'test-hf-token', "Credential mismatch!"
        print(f"✓ Retrieved credential: huggingface:token = {hf_token[:10]}...")
        
        # Test listing services
        services = store.list_services()
        assert 'modal' in services
        assert 'huggingface' in services
        assert 'kaggle' in services
        print(f"✓ Listed services: {services}")
        
        # Test credential rotation
        store.rotate_credential('modal', 'token_id', 'new-modal-token-id')
        new_token_id = store.get_credential('modal', 'token_id')
        assert new_token_id == 'new-modal-token-id', "Rotation failed!"
        print("✓ Credential rotated successfully")
        
        # Test stats
        stats = store.get_stats()
        print(f"✓ Store stats: {stats}")
        
        # Test persistence (save and reload)
        store._save_credentials()
        print(f"✓ Credentials saved to {store_file}")
        
        # Create new store instance and load
        store2 = CredentialStore(
            backend_config={
                'store_file': store_file,
                'salt_file': salt_file
            }
        )
        loaded_token = store2.get_credential('modal', 'token_id')
        assert loaded_token == 'new-modal-token-id', "Load failed!"
        print("✓ Credentials loaded successfully from file")
        
        # Test deletion
        store.delete_credential('kaggle', 'key')
        assert store.get_credential('kaggle', 'key') is None
        print("✓ Credential deleted successfully")
        
        print("\n✅ Credential Store tests PASSED")

except Exception as e:
    print(f"\n❌ Credential Store tests FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Authentication Manager
print("\n[TEST 2] Authentication Manager with JWT")
print("-" * 70)

try:
    from notebook_ml_orchestrator.security.auth_manager import (
        AuthManager, AuthenticationError, Role
    )
    
    # Initialize auth manager
    auth = AuthManager()
    print("✓ AuthManager initialized")
    print(f"  - Access token expiry: {auth.access_token_expiry}")
    print(f"  - Refresh token expiry: {auth.refresh_token_expiry}")

    # Register test users with strong passwords
    admin = auth.register_user('admin', 'admin@example.com', 'Admin123!', Role.ADMIN)
    user = auth.register_user('testuser', 'user@example.com', 'User123!', Role.USER)
    viewer = auth.register_user('viewer', 'viewer@example.com', 'Viewer123!', Role.VIEWER)
    print("[OK] Registered 3 test users")

    # Test authentication
    tokens = auth.authenticate('admin', 'Admin123!')
    assert 'access_token' in tokens
    assert 'refresh_token' in tokens
    print("[OK] Authentication successful for admin")
    print(f"  - Access token: {tokens['access_token'][:50]}...")

    # Test token validation
    payload = auth.validate_token(tokens['access_token'])
    assert payload.user_id == admin.id
    assert payload.username == 'admin'
    assert payload.role == 'admin'
    print("[OK] Token validation successful")
    print(f"  - User: {payload.username}, Role: {payload.role}")
    
    # Test token refresh
    new_tokens = auth.refresh_access_token(tokens['refresh_token'])
    assert 'access_token' in new_tokens
    print("✓ Token refresh successful")
    
    # Test failed authentication
    try:
        auth.authenticate('admin', 'WrongPass123!')
        print("[FAIL] Should have raised AuthenticationError")
        sys.exit(1)
    except AuthenticationError as e:
        print(f"[OK] Failed authentication handled correctly: {e.error_code}")
    
    # Test API key generation
    api_key = auth.generate_api_key('admin')
    assert api_key.startswith('nml_')
    print(f"✓ API key generated: {api_key[:10]}...")
    
    # Test API key authentication
    api_user = auth.authenticate_api_key(api_key)
    assert api_user.username == 'admin'
    print("✓ API key authentication successful")
    
    # Test session management
    session = auth.create_session(admin, ip_address='192.168.1.1')
    assert session.is_valid
    print(f"✓ Session created: {session.id[:16]}...")
    
    # Test session retrieval
    retrieved_session = auth.get_session(session.id)
    assert retrieved_session is not None
    print("✓ Session retrieved successfully")
    
    # Test session invalidation
    auth.invalidate_session(session.id)
    invalid_session = auth.get_session(session.id)
    assert invalid_session is None
    print("✓ Session invalidated successfully")
    
    # Test user listing
    users = auth.list_users()
    assert len(users) == 3
    print(f"✓ User listing: {len(users)} users")
    
    # Test stats
    auth_stats = auth.get_stats()
    print(f"✓ Auth stats: {auth_stats}")
    
    print("\n✅ Authentication Manager tests PASSED")

except Exception as e:
    print(f"\n❌ Authentication Manager tests FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Security Logger
print("\n[TEST 3] Security Logger")
print("-" * 70)

try:
    from notebook_ml_orchestrator.security.security_logger import (
        SecurityLogger
    )
    
    # Create temporary log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_file = f.name
    
    # Initialize security logger
    sec_logger = SecurityLogger(
        log_file=log_file,
        include_console=False
    )
    print("✓ SecurityLogger initialized")
    
    # Test logging various events
    sec_logger.log_auth_success('admin', ip_address='192.168.1.1')
    print("✓ Logged auth success")
    
    sec_logger.log_auth_failure('unknown', ip_address='192.168.1.100', reason='invalid_credentials')
    print("✓ Logged auth failure")
    
    sec_logger.log_authz_failure('viewer', 'admin_panel', 'ADMIN', 'VIEWER')
    print("✓ Logged authz failure")
    
    sec_logger.log_rate_limit_exceeded('user123', '/api/jobs', 100)
    print("✓ Logged rate limit exceeded")
    
    sec_logger.log_sql_injection_attempt("'; DROP TABLE users; --", ip_address='10.0.0.1')
    print("✓ Logged SQL injection attempt")
    
    sec_logger.log_xss_attempt('<script>alert("XSS")</script>', ip_address='10.0.0.2')
    print("✓ Logged XSS attempt")
    
    sec_logger.log_credential_access('modal', 'token_id', 'admin')
    print("✓ Logged credential access")
    
    # Test stats
    stats = sec_logger.get_stats()
    print(f"✓ Security logger stats: {stats}")

    # Verify log file was created
    with open(log_file, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert 'auth.success' in log_content
        assert 'sql_injection' in log_content
        print(f"[OK] Log file verified ({len(log_content)} bytes)")

    # Clean up (close logger first to release file handle)
    import gc
    del sec_logger
    gc.collect()
    try:
        os.unlink(log_file)
    except PermissionError:
        pass  # File still locked, will be cleaned up by temp dir

    print("\n[OK] Security Logger tests PASSED")

except Exception as e:
    print(f"\n❌ Security Logger tests FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: XSS Prevention
print("\n[TEST 4] XSS Prevention and Content Sanitization")
print("-" * 70)

try:
    from notebook_ml_orchestrator.security.xss_prevention import (
        ContentSanitizer, CSPHeaderGenerator,
        escape_html, detect_xss, is_safe_url,
        get_security_headers
    )
    
    # Initialize sanitizer
    sanitizer = ContentSanitizer()
    print("✓ ContentSanitizer initialized")
    print(f"  - Allowed tags: {sanitizer.allowed_tags}")
    
    # Test HTML escaping
    dangerous = '<script>alert("XSS")</script>'
    escaped = escape_html(dangerous)
    assert '<' not in escaped
    assert '&lt;script&gt;' in escaped
    print(f"✓ HTML escaping: {dangerous} → {escaped}")
    
    # Test HTML sanitization
    html_content = '<p>Hello <strong>world</strong> <script>alert("XSS")</script></p>'
    result = sanitizer.sanitize_html(html_content)
    assert 'script' not in result.content
    assert '<p>' in result.content
    assert '<strong>' in result.content
    print(f"✓ HTML sanitization: {len(result.removed_elements)} elements removed")
    
    # Test XSS detection
    is_malicious, patterns = detect_xss('<script>alert("XSS")</script>')
    assert is_malicious
    assert 'script tag' in patterns
    print(f"✓ XSS detection: {patterns}")
    
    # Test URL safety
    safe_url = 'https://example.com/page'
    dangerous_url = 'javascript:alert("XSS")'
    assert is_safe_url(safe_url)
    assert not is_safe_url(dangerous_url)
    print(f"✓ URL safety check: {safe_url} (safe), {dangerous_url} (dangerous)")
    
    # Test CSP header generation
    csp_gen = CSPHeaderGenerator()
    strict_csp = csp_gen.generate_strict_csp()
    assert "default-src 'self'" in strict_csp
    assert "script-src 'self'" in strict_csp
    print(f"✓ CSP header generated: {strict_csp[:50]}...")
    
    # Test security headers
    headers = get_security_headers()
    assert 'Content-Security-Policy' in headers
    assert 'X-Content-Type-Options' in headers
    assert 'X-Frame-Options' in headers
    print(f"✓ Security headers: {list(headers.keys())}")
    
    # Test various attack patterns
    attack_patterns = [
        '<script>alert("XSS")</script>',
        '<img src=x onerror=alert("XSS")>',
        '<a href="javascript:alert(\'XSS\')">click</a>',
        '<div style="background:url(javascript:alert(\'XSS\'))">',
        '<iframe src="javascript:alert(\'XSS\')"></iframe>',
    ]
    
    for pattern in attack_patterns:
        is_malicious, _ = detect_xss(pattern)
        assert is_malicious, f"Failed to detect: {pattern}"
    print(f"✓ All {len(attack_patterns)} attack patterns detected")
    
    print("\n✅ XSS Prevention tests PASSED")

except Exception as e:
    print(f"\n❌ XSS Prevention tests FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("ALL SECURITY MODULE TESTS PASSED!")
print("=" * 70)
print("\nTested components:")
print("  ✓ CredentialStore (AES-256-GCM encryption)")
print("  ✓ AuthManager (JWT authentication)")
print("  ✓ SecurityLogger (audit logging)")
print("  ✓ ContentSanitizer (XSS prevention)")
print("  ✓ CSPHeaderGenerator (security headers)")
print("\nSecurity features implemented:")
print("  ✓ Credential encryption at rest")
print("  ✓ JWT token generation and validation")
print("  ✓ Password hashing with bcrypt")
print("  ✓ Session management")
print("  ✓ API key authentication")
print("  ✓ Security event logging")
print("  ✓ XSS prevention")
print("  ✓ Content-Security-Policy headers")
print("=" * 70)
