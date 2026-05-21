/**
 * Agent Memory Toolkit - Dashboard JavaScript
 * 
 * Handles data fetching, chart rendering, and UI updates.
 */

// Chart instances
let memoriesChart = null;
let domainsChart = null;
let searchesChart = null;
let storageChart = null;
let branchesChart = null;

// Chart.js default configuration
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
Chart.defaults.color = '#64748b';

// Color palette
const colors = {
    primary: '#6366f1',
    secondary: '#8b5cf6',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    info: '#06b6d4',
    gray: '#94a3b8',
    palette: [
        '#6366f1', '#8b5cf6', '#a855f7', '#d946ef',
        '#ec4899', '#f43f5e', '#ef4444', '#f97316',
        '#f59e0b', '#eab308', '#84cc16', '#22c55e',
        '#10b981', '#14b8a6', '#06b6d4', '#0ea5e9'
    ]
};

/**
 * Format bytes to human-readable string
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Format number with commas
 */
function formatNumber(num) {
    if (num === null || num === undefined) return '-';
    return num.toLocaleString();
}

/**
 * Format date for display
 */
function formatDate(dateStr) {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

/**
 * Format datetime for display
 */
function formatDateTime(dateStr) {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Fetch analytics data from API
 */
async function fetchAnalytics(days = 30) {
    try {
        const response = await fetch(`/api/stats?days=${days}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch analytics:', error);
        return null;
    }
}

/**
 * Update stat cards with data
 */
function updateStatCards(data) {
    const stats = data.memory_stats;
    const searches = data.search_trends;
    const storage = data.storage_metrics;
    
    // Total memories
    document.getElementById('total-memories').textContent = formatNumber(stats.total_memories);
    document.getElementById('active-memories').textContent = formatNumber(stats.active_memories);
    document.getElementById('deleted-memories').textContent = formatNumber(stats.deleted_memories);
    
    // Branches
    document.getElementById('total-branches').textContent = formatNumber(stats.total_branches);
    document.getElementById('total-commits').textContent = formatNumber(stats.total_commits);
    
    // Searches
    document.getElementById('searches-today').textContent = formatNumber(searches.searches_today);
    document.getElementById('searches-week').textContent = formatNumber(searches.searches_this_week);
    
    // Storage
    document.getElementById('storage-size').textContent = formatBytes(storage.total_size_bytes);
    document.getElementById('avg-memory-size').textContent = formatBytes(stats.avg_memory_size);
    
    // Generated at
    document.getElementById('generated-at').textContent = formatDateTime(data.generated_at);
}

/**
 * Create/update memories over time chart
 */
function updateMemoriesChart(data) {
    const ctx = document.getElementById('memories-chart').getContext('2d');
    const chartData = data.memory_stats.memories_by_day;
    
    const labels = chartData.map(d => d.timestamp);
    const values = chartData.map(d => d.value);
    
    // Calculate cumulative values
    let cumulative = 0;
    const cumulativeValues = values.map(v => cumulative += v);
    
    if (memoriesChart) {
        memoriesChart.destroy();
    }
    
    memoriesChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Daily New Memories',
                    data: values,
                    borderColor: colors.primary,
                    backgroundColor: colors.primary + '20',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    yAxisID: 'y'
                },
                {
                    label: 'Cumulative Total',
                    data: cumulativeValues,
                    borderColor: colors.secondary,
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0.4,
                    pointRadius: 0,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        title: function(items) {
                            return formatDate(items[0].label);
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM d'
                        }
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Daily Count'
                    },
                    grid: {
                        color: '#e2e8f0'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Cumulative'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

/**
 * Create/update domain distribution pie chart
 */
function updateDomainsChart(data) {
    const ctx = document.getElementById('domains-chart').getContext('2d');
    const distribution = data.domain_distribution;
    
    const labels = Object.keys(distribution.domain_counts);
    const values = Object.values(distribution.domain_counts);
    
    if (domainsChart) {
        domainsChart.destroy();
    }
    
    if (labels.length === 0) {
        // Show no data message
        ctx.canvas.parentElement.innerHTML = `
            <div class="no-data">
                <div class="no-data-icon">📊</div>
                <div>No domain data available</div>
            </div>
        `;
        return;
    }
    
    domainsChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [{
                data: values,
                backgroundColor: colors.palette.slice(0, labels.length),
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${context.label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            },
            cutout: '60%'
        }
    });
}

/**
 * Create/update search trends chart
 */
function updateSearchesChart(data) {
    const ctx = document.getElementById('searches-chart').getContext('2d');
    const chartData = data.search_trends.searches_by_day;
    
    const labels = chartData.map(d => d.timestamp);
    const values = chartData.map(d => d.value);
    
    if (searchesChart) {
        searchesChart.destroy();
    }
    
    searchesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Search Queries',
                data: values,
                backgroundColor: colors.info,
                borderRadius: 4,
                barThickness: 'flex',
                maxBarThickness: 30
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        title: function(items) {
                            return formatDate(items[0].label);
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM d'
                        }
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#e2e8f0'
                    }
                }
            }
        }
    });
}

/**
 * Update top queries list
 */
function updateTopQueries(data) {
    const list = document.getElementById('top-queries-list');
    const topQueries = data.search_trends.top_queries;
    
    if (topQueries.length === 0) {
        list.innerHTML = `
            <li class="no-data">
                <div class="no-data-icon">🔍</div>
                <div>No search queries recorded</div>
            </li>
        `;
        return;
    }
    
    list.innerHTML = topQueries.map(q => `
        <li>
            <span class="query-text" title="${q.query}">${q.query}</span>
            <span class="query-count">${q.count}</span>
        </li>
    `).join('');
}

/**
 * Create/update storage breakdown chart
 */
function updateStorageChart(data) {
    const ctx = document.getElementById('storage-chart').getContext('2d');
    const storage = data.storage_metrics;
    
    const labels = ['FTS Index', 'Embeddings', 'Metadata', 'Other'];
    const values = [
        storage.fts_index_size_bytes,
        storage.embeddings_size_bytes,
        storage.metadata_size_bytes,
        Math.max(0, storage.database_size_bytes - storage.fts_index_size_bytes - 
                 storage.embeddings_size_bytes - storage.metadata_size_bytes)
    ];
    
    if (storageChart) {
        storageChart.destroy();
    }
    
    storageChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [
                    colors.primary,
                    colors.success,
                    colors.warning,
                    colors.gray
                ],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${formatBytes(context.raw)}`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create/update branches comparison chart
 */
function updateBranchesChart(data) {
    const ctx = document.getElementById('branches-chart').getContext('2d');
    const branches = data.branch_comparison.branches;
    
    if (branches.length === 0) {
        ctx.canvas.parentElement.innerHTML = `
            <div class="no-data">
                <div class="no-data-icon">🌿</div>
                <div>No branches found</div>
            </div>
        `;
        return;
    }
    
    const labels = branches.map(b => b.name);
    const memoryData = branches.map(b => b.memory_count);
    const commitData = branches.map(b => b.commit_count);
    
    if (branchesChart) {
        branchesChart.destroy();
    }
    
    branchesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Memories',
                    data: memoryData,
                    backgroundColor: colors.primary,
                    borderRadius: 4
                },
                {
                    label: 'Commits',
                    data: commitData,
                    backgroundColor: colors.secondary,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#e2e8f0'
                    }
                }
            }
        }
    });
}

/**
 * Update branches table
 */
function updateBranchesTable(data) {
    const tbody = document.getElementById('branches-table-body');
    const branches = data.branch_comparison.branches;
    
    if (branches.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="no-data">No branches found</td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = branches.map(b => `
        <tr>
            <td><strong>${b.name}</strong></td>
            <td>${formatNumber(b.memory_count)}</td>
            <td>${formatNumber(b.commit_count)}</td>
            <td>${formatDate(b.created_at)}</td>
            <td>${formatDateTime(b.last_commit)}</td>
            <td>
                <span class="branch-badge ${b.is_current ? 'current' : 'inactive'}">
                    ${b.is_current ? '✓ Current' : 'Inactive'}
                </span>
            </td>
        </tr>
    `).join('');
}

/**
 * Refresh all dashboard data
 */
async function refreshDashboard() {
    const refreshBtn = document.getElementById('refresh-btn');
    const timeRange = document.getElementById('time-range').value;
    
    refreshBtn.classList.add('loading');
    refreshBtn.textContent = '⏳ Loading...';
    
    try {
        const data = await fetchAnalytics(parseInt(timeRange));
        
        if (data && !data.error) {
            updateStatCards(data);
            updateMemoriesChart(data);
            updateDomainsChart(data);
            updateSearchesChart(data);
            updateTopQueries(data);
            updateStorageChart(data);
            updateBranchesChart(data);
            updateBranchesTable(data);
        } else {
            console.error('Failed to load analytics data:', data?.error);
        }
    } catch (error) {
        console.error('Error refreshing dashboard:', error);
    } finally {
        refreshBtn.classList.remove('loading');
        refreshBtn.textContent = '🔄 Refresh';
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initial load
    refreshDashboard();
    
    // Refresh button
    document.getElementById('refresh-btn').addEventListener('click', refreshDashboard);
    
    // Time range change
    document.getElementById('time-range').addEventListener('change', refreshDashboard);
    
    // Auto-refresh every 60 seconds
    setInterval(refreshDashboard, 60000);
});
