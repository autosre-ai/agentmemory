"""Access control for Team Memory Protocol."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from .models import Permission, AccessRule
from .exceptions import PermissionDeniedError


class AccessControl:
    """
    Manages access control for team memories.
    
    Supports:
    - Per-agent permissions
    - Per-namespace permissions
    - Combined agent + namespace rules
    - Admin role for managing access
    """
    
    def __init__(self, conn: sqlite3.Connection, owner_agent_id: str | None = None):
        """
        Initialize access control.
        
        Args:
            conn: SQLite database connection
            owner_agent_id: Agent ID that owns this store (has full admin access)
        """
        self._conn = conn
        self._owner_agent_id = owner_agent_id
        self._cache: dict[str, dict[str | None, Permission]] = {}
    
    def grant(
        self,
        agent_id: str | None,
        permission: Permission,
        namespace: str | None = None,
        granted_by: str | None = None,
    ) -> AccessRule:
        """
        Grant permission to an agent.
        
        Args:
            agent_id: Agent to grant permission to (None = all agents)
            permission: Permission level to grant
            namespace: Namespace to grant access to (None = all namespaces)
            granted_by: Agent granting the permission
            
        Returns:
            The created AccessRule
        """
        # Check if granting agent has admin permission
        if granted_by and granted_by != self._owner_agent_id:
            if not self.has_permission(granted_by, Permission.ADMIN, namespace):
                raise PermissionDeniedError(granted_by, "grant permissions", namespace)
        
        rule = AccessRule(
            agent_id=agent_id,
            namespace=namespace,
            permission=permission,
            created_at=datetime.utcnow(),
        )
        
        self._conn.execute(
            """
            INSERT OR REPLACE INTO team_access_rules 
            (agent_id, namespace, permission, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (agent_id, namespace, permission.value, rule.created_at.isoformat()),
        )
        self._conn.commit()
        
        # Invalidate cache
        if agent_id in self._cache:
            del self._cache[agent_id]
        
        return rule
    
    def revoke(
        self,
        agent_id: str | None,
        namespace: str | None = None,
        revoked_by: str | None = None,
    ) -> bool:
        """
        Revoke all permissions for an agent.
        
        Args:
            agent_id: Agent to revoke permissions from
            namespace: Namespace to revoke access from (None = all namespaces)
            revoked_by: Agent revoking the permission
            
        Returns:
            True if permissions were revoked
        """
        if revoked_by and revoked_by != self._owner_agent_id:
            if not self.has_permission(revoked_by, Permission.ADMIN, namespace):
                raise PermissionDeniedError(revoked_by, "revoke permissions", namespace)
        
        if namespace:
            cursor = self._conn.execute(
                "DELETE FROM team_access_rules WHERE agent_id = ? AND namespace = ?",
                (agent_id, namespace),
            )
        else:
            cursor = self._conn.execute(
                "DELETE FROM team_access_rules WHERE agent_id = ?",
                (agent_id,),
            )
        
        self._conn.commit()
        
        # Invalidate cache
        if agent_id in self._cache:
            del self._cache[agent_id]
        
        return cursor.rowcount > 0
    
    def get_permission(
        self,
        agent_id: str,
        namespace: str | None = None,
    ) -> Permission:
        """
        Get the effective permission for an agent.
        
        Permission resolution order:
        1. Owner always has ADMIN
        2. Specific agent + specific namespace rule
        3. Specific agent + all namespaces rule
        4. All agents + specific namespace rule
        5. All agents + all namespaces rule
        6. Default: NONE
        
        Args:
            agent_id: Agent to check
            namespace: Namespace to check (None = global)
            
        Returns:
            The effective Permission level
        """
        # Owner has full access
        if agent_id == self._owner_agent_id:
            return Permission.ADMIN
        
        # Check cache
        if agent_id in self._cache and namespace in self._cache[agent_id]:
            return self._cache[agent_id][namespace]
        
        # Query for all applicable rules
        cursor = self._conn.execute(
            """
            SELECT agent_id, namespace, permission FROM team_access_rules
            WHERE (agent_id = ? OR agent_id IS NULL)
              AND (namespace = ? OR namespace IS NULL)
            ORDER BY 
                CASE WHEN agent_id IS NOT NULL THEN 0 ELSE 1 END,
                CASE WHEN namespace IS NOT NULL THEN 0 ELSE 1 END
            LIMIT 1
            """,
            (agent_id, namespace),
        )
        
        row = cursor.fetchone()
        if row:
            permission = Permission(row[2])
        else:
            permission = Permission.NONE
        
        # Update cache
        if agent_id not in self._cache:
            self._cache[agent_id] = {}
        self._cache[agent_id][namespace] = permission
        
        return permission
    
    def has_permission(
        self,
        agent_id: str,
        required: Permission,
        namespace: str | None = None,
    ) -> bool:
        """
        Check if an agent has at least the required permission level.
        
        Args:
            agent_id: Agent to check
            required: Required permission level
            namespace: Namespace to check
            
        Returns:
            True if agent has sufficient permission
        """
        actual = self.get_permission(agent_id, namespace)
        return actual.value >= required.value
    
    def check_permission(
        self,
        agent_id: str,
        required: Permission,
        namespace: str | None = None,
        operation: str = "access",
    ) -> None:
        """
        Check permission and raise if insufficient.
        
        Args:
            agent_id: Agent to check
            required: Required permission level
            namespace: Namespace to check
            operation: Description of operation for error message
            
        Raises:
            PermissionDeniedError: If agent lacks required permission
        """
        if not self.has_permission(agent_id, required, namespace):
            raise PermissionDeniedError(agent_id, operation, namespace)
    
    def list_rules(
        self,
        agent_id: str | None = None,
        namespace: str | None = None,
    ) -> list[AccessRule]:
        """
        List access rules.
        
        Args:
            agent_id: Filter by agent (None = all)
            namespace: Filter by namespace (None = all)
            
        Returns:
            List of AccessRule objects
        """
        query = "SELECT agent_id, namespace, permission, created_at FROM team_access_rules"
        params: list[Any] = []
        conditions: list[str] = []
        
        if agent_id is not None:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        
        if namespace is not None:
            conditions.append("namespace = ?")
            params.append(namespace)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor = self._conn.execute(query, params)
        return [AccessRule.from_dict({
            "agent_id": row[0],
            "namespace": row[1],
            "permission": row[2],
            "created_at": row[3],
        }) for row in cursor.fetchall()]
    
    def clear_cache(self) -> None:
        """Clear the permission cache."""
        self._cache.clear()
