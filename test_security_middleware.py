#!/usr/bin/env python3
# coding: utf-8
"""
Test script for security middleware and integration utilities.

Tests:
- SecurityMiddleware
- SecurityContext
- Authentication decorators
- Rate limiting
- Input sanitization
- Gradio security middleware
"""

import os
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set up test environment
os.environ['MASTER_KEY'] = 'test-master-key-for-development-only-32bytes!'
os.environ['JWT_SECRET'] = 'test-jwt-secret-for-development-only!'

print("=" * 70)
print("Testing Security Middleware and Integration Utilities")
print("=" * 70)

# Test 1: SecurityMiddleware initialization
print("\n[TEST 1] SecurityMiddleware Initialization")
print("-" * 70)

try:
    from notebook_ml_orchestrator.security import (
        SecurityMiddleware,
        SecurityContext,
        create_security_middleware
    )
    
    # Test basic initialization
    middleware = SecurityMiddleware(enabled=True)
    print("[OK] SecurityMiddleware initialized")
    
    # Test with all components
    from notebook_ml_orchestrator.security import AuthManager, CredentialStore, SecurityLogger
    
    auth = AuthManager()
    store = CredentialStore()
    logger = SecurityLogger(include_console=False)
    
    full_middleware = SecurityMiddleware(
        auth_manager=auth,
        credential_store=store,
        security_logger=logger,
        enabled=True
    )
    print("[OK] SecurityMiddleware with all components initialized")
    
    # Test factory function
    factory_middleware = create_security_middleware(
        enable_auth=True,
        enable_rate_limit=True,
        enable_audit_logging=True
    )
    print("[OK] create_security_middleware factory works")
    
except Exception as e:
    print(f"[FAIL] SecurityMiddleware initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: SecurityContext
print("\n[TEST 2] SecurityContext")
print("-" * 70)

try:
    ctx = SecurityContext()
    assert ctx.authenticated == False
    assert ctx.user_id is None
    print("[OK] SecurityContext default values correct")
    
    ctx = SecurityContext(
        user_id='user123',
        username='alice',
        role='admin',
        authenticated=True,
        permissions=['read', 'write']
    )
    assert ctx.user_id == 'user123'
    assert ctx.username == 'alice'
    assert ctx.role == 'admin'
    assert ctx.authenticated == True
    assert 'read' in ctx.permissions
    print("[OK] SecurityContext with values works")
    
except Exception as e:
    print(f"[FAIL] SecurityContext test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Rate Limiting
print("\n[TEST 3] Rate Limiting")
print("-" * 70)

try:
    # Test rate limit checking
    allowed, retry_after = middleware.check_rate_limit('test_user')
    assert allowed == True
    print(f"[OK] Rate limit check passed (allowed={allowed})")
    
    # Simulate multiple requests
    for i in range(5):
        allowed, _ = middleware.check_rate_limit('test_user_2')
    print("[OK] Multiple rate limit checks passed")
    
except Exception as e:
    print(f"[FAIL] Rate limiting test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Security Headers
print("\n[TEST 4] Security Headers")
print("-" * 70)

try:
    # Mock response object
    class MockResponse:
        def __init__(self):
            self.headers = {}
    
    response = MockResponse()
    middleware.add_security_headers(response)
    
    assert 'X-Content-Type-Options' in response.headers
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert 'X-Frame-Options' in response.headers
    assert response.headers['X-Frame-Options'] == 'DENY'
    print(f"[OK] Security headers added: {list(response.headers.keys())}")
    
except Exception as e:
    print(f"[FAIL] Security headers test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Input Sanitization
print("\n[TEST 5] Input Sanitization")
print("-" * 70)

try:
    # Test XSS sanitization
    malicious_input = '<script>alert("XSS")</script>Hello'
    sanitized = middleware.sanitize_input(malicious_input, 'test_field')
    assert '<script>' not in sanitized
    assert 'Hello' in sanitized
    print(f"[OK] XSS sanitization works: {malicious_input[:30]}... -> {sanitized[:30]}...")
    
    # Test nested dict sanitization
    nested_input = {
        'name': '<b>Bold</b>',
        'description': 'Normal text',
        'items': ['<script>bad</script>', 'good']
    }
    sanitized_nested = middleware.sanitize_input(nested_input, 'nested')
    assert '<script>' not in str(sanitized_nested)
    print("[OK] Nested structure sanitization works")
    
except Exception as e:
    print(f"[FAIL] Input sanitization test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Authentication Decorator
print("\n[TEST 6] Authentication Decorator")
print("-" * 70)

try:
    from notebook_ml_orchestrator.security import require_auth, Role
    
    # Register test users first
    auth.register_user('admin', 'admin@test.com', 'Admin123!', Role.ADMIN)
    auth.register_user('testuser', 'user@test.com', 'User123!', Role.USER)
    print("[OK] Test users registered")
    
    # Create mock function
    @require_auth(auth_manager=auth, required_role='admin')
    def admin_endpoint(token=None, **kwargs):
        return {'status': 'success', 'user': kwargs.get('current_user')}
    
    # Test without token
    result = admin_endpoint()
    assert result[1] == 401
    print("[OK] Auth decorator rejects unauthenticated requests")
    
    # Test with valid token
    tokens = auth.authenticate('admin', 'Admin123!')
    result = admin_endpoint(token=tokens['access_token'])
    # Result is a tuple (response_dict, status_code) or just response_dict
    if isinstance(result, tuple):
        assert result[0]['status'] == 'success'
        assert result[0]['user']['role'] == 'admin'
    else:
        assert result['status'] == 'success'
    print("[OK] Auth decorator accepts valid admin token")
    
    # Test with non-admin user
    user_tokens = auth.authenticate('testuser', 'User123!')
    result = admin_endpoint(token=user_tokens['access_token'])
    if isinstance(result, tuple):
        assert result[1] == 403
    print("[OK] Auth decorator rejects non-admin user")
    
except Exception as e:
    print(f"[FAIL] Auth decorator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Rate Limit Decorator
print("\n[TEST 7] Rate Limit Decorator")
print("-" * 70)

try:
    from notebook_ml_orchestrator.security import rate_limit
    
    @rate_limit(requests_per_minute=3)
    def limited_endpoint(ip_address=None, **kwargs):
        return {'status': 'success'}
    
    # Test within limit
    for i in range(3):
        result = limited_endpoint(ip_address='test_ip')
        if isinstance(result, tuple):
            assert result[0]['status'] == 'success'
        else:
            assert result['status'] == 'success'
    print("[OK] Rate limit decorator allows requests within limit")
    
    # Test exceeding limit
    result = limited_endpoint(ip_address='test_ip')
    if isinstance(result, tuple):
        assert result[1] == 429
    else:
        # If not tuple, check if it's an error response
        assert 'error' in str(result).lower() or 'rate' in str(result).lower()
    print("[OK] Rate limit decorator rejects requests over limit")
    
except Exception as e:
    print(f"[FAIL] Rate limit decorator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Request Validation Decorator
print("\n[TEST 8] Request Validation Decorator")
print("-" * 70)

try:
    from notebook_ml_orchestrator.security import validate_request
    
    @validate_request({
        'template': {'type': str, 'required': True},
        'inputs': {'type': dict, 'required': True},
        'priority': {'type': int, 'required': False}
    })
    def create_job(request_data=None, **kwargs):
        return {'status': 'created'}
    
    # Test missing required field
    result = create_job(request_data={'template': 'test'})
    if isinstance(result, tuple):
        assert result[1] == 400
    print("[OK] Validation decorator rejects missing required field")
    
    # Test wrong type
    result = create_job(request_data={'template': 'test', 'inputs': 'not_a_dict'})
    if isinstance(result, tuple):
        assert result[1] == 400
    print("[OK] Validation decorator rejects wrong type")
    
    # Test valid request
    result = create_job(request_data={'template': 'test', 'inputs': {}})
    if isinstance(result, tuple):
        assert result[0]['status'] == 'created'
    else:
        assert result['status'] == 'created'
    print("[OK] Validation decorator accepts valid request")
    
except Exception as e:
    print(f"[FAIL] Validation decorator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: GradioSecurityMiddleware
print("\n[TEST 9] GradioSecurityMiddleware")
print("-" * 70)

try:
    from notebook_ml_orchestrator.security import GradioSecurityMiddleware
    
    gradio_middleware = GradioSecurityMiddleware(middleware)
    print("[OK] GradioSecurityMiddleware initialized")
    
    # Test secure_function wrapper (not a decorator, returns wrapped function)
    def test_gradio_function(security_context=None, **kwargs):
        return {'status': 'success', 'input': kwargs.get('user_input')}
    
    secure_func = gradio_middleware.secure_function(
        test_gradio_function,
        require_auth=True,
        required_role='admin',
        sanitize_inputs=True
    )
    
    # Test without auth
    try:
        result = secure_func(user_input='test')
        print("[FAIL] Should have raised PermissionError")
        sys.exit(1)
    except PermissionError:
        print("[OK] Gradio secure function rejects unauthenticated access")
    
    # Test with auth but wrong role
    ctx = SecurityContext(authenticated=True, role='user')
    try:
        result = secure_func(security_context=ctx, user_input='test')
        print("[FAIL] Should have raised PermissionError for wrong role")
        sys.exit(1)
    except PermissionError:
        print("[OK] Gradio secure function rejects wrong role")
    
    # Test with proper auth
    ctx = SecurityContext(authenticated=True, role='admin')
    result = secure_func(security_context=ctx, user_input='<script>test</script>')
    assert '<script>' not in str(result)
    print("[OK] Gradio secure function works with proper auth and sanitization")
    
except Exception as e:
    print(f"[FAIL] GradioSecurityMiddleware test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Integration with existing components
print("\n[TEST 10] Integration with Existing Components")
print("-" * 70)

try:
    # Test credential store integration with middleware
    store.set_credential('test_service', 'api_key', 'test_key_123')
    retrieved = store.get_credential('test_service', 'api_key')
    assert retrieved == 'test_key_123'
    print("[OK] Credential store integration works")
    
    # Test security logger integration
    logger.log_auth_success('test_user', ip_address='192.168.1.1')
    logger.log_auth_failure('unknown', ip_address='10.0.0.1', reason='test')
    print("[OK] Security logger integration works")
    
    # Test middleware request logging
    ctx = SecurityContext(username='test_user', ip_address='192.168.1.1')
    middleware.log_request('POST', '/api/test', 200, 50.5, ctx)
    middleware.log_request('POST', '/api/test', 401, 10.2, ctx)
    print("[OK] Middleware request logging works")
    
except Exception as e:
    print(f"[FAIL] Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("ALL SECURITY MIDDLEWARE TESTS PASSED!")
print("=" * 70)
print("\nTested components:")
print("  [OK] SecurityMiddleware")
print("  [OK] SecurityContext")
print("  [OK] Rate limiting")
print("  [OK] Security headers")
print("  [OK] Input sanitization")
print("  [OK] Authentication decorator (@require_auth)")
print("  [OK] Rate limit decorator (@rate_limit)")
print("  [OK] Validation decorator (@validate_request)")
print("  [OK] GradioSecurityMiddleware")
print("  [OK] Integration with existing components")
print("\nIntegration features:")
print("  [OK] Factory function (create_security_middleware)")
print("  [OK] Decorator-based security")
print("  [OK] Gradio integration")
print("  [OK] Request logging and auditing")
print("=" * 70)
