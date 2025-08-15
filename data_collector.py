#!/usr/bin/env python3
"""
data_collector.py - Bug Bounty Reconnaissance Data Collector

This module provides safe and ethical data collection capabilities for bug bounty
reconnaissance activities. It focuses on gathering publicly available information
and verifying live endpoints through controlled HTTP probing.

Author: Security Research Tool
License: MIT
"""

import os
import sys
import json
import time
import socket
import requests
import threading
from pathlib import Path
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import logging utilities
try:
    from utils import setup_logger, log_info, log_warning, log_error
except ImportError:
    # Fallback logging if utils.py is not available
    import logging
    
    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def log_info(logger, msg): logger.info(msg)
    def log_warning(logger, msg): logger.warning(msg)
    def log_error(logger, msg): logger.error(msg)


@dataclass
class EndpointInfo:
    """Data class to store endpoint information"""
    url: str
    status_code: Optional[int] = None
    title: str = ""
    server: str = ""
    content_type: str = ""
    response_time: float = 0.0
    content_length: int = 0
    is_live: bool = False
    technologies: List[str] = None
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.technologies is None:
            self.technologies = []
        if self.headers is None:
            self.headers = {}


@dataclass
class ReconData:
    """Main data structure for reconnaissance results"""
    target: str
    subdomains: Set[str]
    resolved_ips: Dict[str, str]
    live_endpoints: List[EndpointInfo]
    total_endpoints: int
    scan_timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'target': self.target,
            'subdomains': list(self.subdomains),
            'resolved_ips': self.resolved_ips,
            'live_endpoints': [asdict(endpoint) for endpoint in self.live_endpoints],
            'total_endpoints': self.total_endpoints,
            'scan_timestamp': self.scan_timestamp
        }


class DataCollector:
    """Main class for collecting bug bounty reconnaissance data"""
    
    def __init__(self, target_name: str = "unknown", max_workers: int = 20, 
                 timeout: int = 10, rate_limit: float = 0.1):
        """
        Initialize the DataCollector
        
        Args:
            target_name: Name of the target being scanned
            max_workers: Maximum number of concurrent threads
            timeout: HTTP request timeout in seconds
            rate_limit: Delay between requests in seconds
        """
        self.target_name = target_name
        self.max_workers = max_workers
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.logger = setup_logger(f"DataCollector-{target_name}")
        
        # User agent for ethical reconnaissance
        self.user_agent = "Bug-Bounty-Research-Tool/1.0 (Ethical Security Testing)"
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        
        log_info(self.logger, f"DataCollector initialized for target: {target_name}")
    
    def read_file_lines(self, filepath: str) -> List[str]:
        """
        Read lines from a file, handling common formats
        
        Args:
            filepath: Path to the input file
            
        Returns:
            List of cleaned lines from the file
        """
        try:
            path = Path(filepath)
            if not path.exists():
                log_warning(self.logger, f"File not found: {filepath}")
                return []
            
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f.readlines() 
                        if line.strip() and not line.startswith('#')]
            
            log_info(self.logger, f"Read {len(lines)} lines from {filepath}")
            return lines
            
        except Exception as e:
            log_error(self.logger, f"Error reading file {filepath}: {e}")
            return []
    
    def load_subdomains(self, filepath: str = "subdomains.txt") -> Set[str]:
        """Load subdomains from file"""
        subdomains = set()
        lines = self.read_file_lines(filepath)
        
        for line in lines:
            # Clean and validate subdomain
            subdomain = line.lower().strip()
            if self._is_valid_domain(subdomain):
                subdomains.add(subdomain)
        
        log_info(self.logger, f"Loaded {len(subdomains)} valid subdomains")
        return subdomains
    
    def load_resolved_ips(self, filepath: str = "resolved.txt") -> Dict[str, str]:
        """
        Load resolved domain-to-IP mappings from file
        Expected format: domain:ip or domain,ip
        """
        resolved = {}
        lines = self.read_file_lines(filepath)
        
        for line in lines:
            try:
                # Handle both : and , separators
                if ':' in line:
                    domain, ip = line.split(':', 1)
                elif ',' in line:
                    domain, ip = line.split(',', 1)
                else:
                    continue
                
                domain = domain.strip().lower()
                ip = ip.strip()
                
                if self._is_valid_domain(domain) and self._is_valid_ip(ip):
                    resolved[domain] = ip
                    
            except Exception as e:
                log_warning(self.logger, f"Error parsing resolved line '{line}': {e}")
        
        log_info(self.logger, f"Loaded {len(resolved)} domain-IP mappings")
        return resolved
    
    def load_urls(self, filepath: str = "urls.txt") -> List[str]:
        """Load URLs from file"""
        urls = []
        lines = self.read_file_lines(filepath)
        
        for line in lines:
            # Ensure URL has a scheme
            if not line.startswith(('http://', 'https://')):
                # Default to https for security
                line = f"https://{line}"
            
            if self._is_valid_url(line):
                urls.append(line)
        
        log_info(self.logger, f"Loaded {len(urls)} valid URLs")
        return urls
    
    def probe_endpoint(self, url: str) -> EndpointInfo:
        """
        Probe a single endpoint safely with rate limiting
        
        Args:
            url: URL to probe
            
        Returns:
            EndpointInfo object with gathered data
        """
        endpoint = EndpointInfo(url=url)
        
        try:
            # Rate limiting
            time.sleep(self.rate_limit)
            
            # Perform HTTP request
            start_time = time.time()
            response = self.session.get(
                url, 
                timeout=self.timeout, 
                allow_redirects=True,
                verify=True  # SSL verification for security
            )
            response_time = time.time() - start_time
            
            # Populate endpoint info
            endpoint.status_code = response.status_code
            endpoint.response_time = round(response_time, 3)
            endpoint.content_length = len(response.content)
            endpoint.is_live = 200 <= response.status_code < 400
            
            # Extract headers
            endpoint.headers = dict(response.headers)
            endpoint.server = response.headers.get('Server', '')
            endpoint.content_type = response.headers.get('Content-Type', '')
            
            # Extract title from HTML
            if 'text/html' in endpoint.content_type.lower():
                endpoint.title = self._extract_title(response.text)
            
            # Detect technologies
            endpoint.technologies = self._detect_technologies(response)
            
            if endpoint.is_live:
                log_info(self.logger, f"Live endpoint: {url} [{endpoint.status_code}]")
            
        except requests.exceptions.SSLError as e:
            log_warning(self.logger, f"SSL error for {url}: {e}")
        except requests.exceptions.Timeout:
            log_warning(self.logger, f"Timeout for {url}")
        except requests.exceptions.ConnectionError as e:
            log_warning(self.logger, f"Connection error for {url}: {e}")
        except Exception as e:
            log_error(self.logger, f"Error probing {url}: {e}")
        
        return endpoint
    
    def probe_endpoints_concurrent(self, urls: List[str]) -> List[EndpointInfo]:
        """
        Probe multiple endpoints concurrently with proper rate limiting
        
        Args:
            urls: List of URLs to probe
            
        Returns:
            List of EndpointInfo objects
        """
        endpoints = []
        
        log_info(self.logger, f"Starting concurrent probing of {len(urls)} endpoints")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {executor.submit(self.probe_endpoint, url): url 
                           for url in urls}
            
            # Collect results
            for future in as_completed(future_to_url):
                try:
                    endpoint = future.result()
                    endpoints.append(endpoint)
                except Exception as e:
                    url = future_to_url[future]
                    log_error(self.logger, f"Exception in thread for {url}: {e}")
        
        live_count = sum(1 for ep in endpoints if ep.is_live)
        log_info(self.logger, f"Probing complete: {live_count}/{len(endpoints)} live endpoints")
        
        return endpoints
    
    def collect_all_data(self, subdomains_file: str = "subdomains.txt",
                        resolved_file: str = "resolved.txt",
                        urls_file: str = "urls.txt") -> ReconData:
        """
        Collect all reconnaissance data from input files
        
        Args:
            subdomains_file: Path to subdomains file
            resolved_file: Path to resolved domains file
            urls_file: Path to URLs file
            
        Returns:
            ReconData object containing all collected information
        """
        log_info(self.logger, "Starting comprehensive data collection")
        
        # Load data from files
        subdomains = self.load_subdomains(subdomains_file)
        resolved_ips = self.load_resolved_ips(resolved_file)
        urls = self.load_urls(urls_file)
        
        # Combine URLs from subdomains and explicit URLs
        all_urls = set(urls)
        
        # Generate URLs from subdomains (both HTTP and HTTPS)
        for subdomain in subdomains:
            all_urls.add(f"https://{subdomain}")
            all_urls.add(f"http://{subdomain}")
        
        all_urls = list(all_urls)
        
        # Probe all endpoints
        live_endpoints = self.probe_endpoints_concurrent(all_urls)
        
        # Create reconnaissance data structure
        recon_data = ReconData(
            target=self.target_name,
            subdomains=subdomains,
            resolved_ips=resolved_ips,
            live_endpoints=live_endpoints,
            total_endpoints=len(all_urls),
            scan_timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        )
        
        log_info(self.logger, f"Data collection complete for {self.target_name}")
        return recon_data
    
    def save_results(self, recon_data: ReconData, output_file: str = "recon_results.json"):
        """Save reconnaissance results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(recon_data.to_dict(), f, indent=2, ensure_ascii=False)
            
            log_info(self.logger, f"Results saved to {output_file}")
            
        except Exception as e:
            log_error(self.logger, f"Error saving results: {e}")
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Validate domain name format"""
        if not domain or len(domain) > 253:
            return False
        
        # Basic domain validation
        parts = domain.split('.')
        if len(parts) < 2:
            return False
        
        for part in parts:
            if not part or len(part) > 63:
                return False
            if not part.replace('-', '').isalnum():
                return False
        
        return True
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format"""
        try:
            socket.inet_aton(ip)
            return True
        except socket.error:
            return False
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _extract_title(self, html_content: str) -> str:
        """Extract page title from HTML content"""
        try:
            import re
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
            if title_match:
                return title_match.group(1).strip()
        except Exception:
            pass
        return ""
    
    def _detect_technologies(self, response) -> List[str]:
        """Detect technologies from HTTP response"""
        technologies = []
        
        try:
            headers = response.headers
            
            # Server detection
            server = headers.get('Server', '').lower()
            if 'nginx' in server:
                technologies.append('Nginx')
            elif 'apache' in server:
                technologies.append('Apache')
            elif 'iis' in server:
                technologies.append('IIS')
            
            # Framework detection
            if 'x-powered-by' in headers:
                powered_by = headers['x-powered-by'].lower()
                if 'php' in powered_by:
                    technologies.append('PHP')
                elif 'asp.net' in powered_by:
                    technologies.append('ASP.NET')
            
            # Content-based detection
            content = response.text.lower()
            if 'wordpress' in content:
                technologies.append('WordPress')
            elif 'drupal' in content:
                technologies.append('Drupal')
            elif 'joomla' in content:
                technologies.append('Joomla')
            
        except Exception:
            pass
        
        return technologies
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources"""
        self.session.close()


# Example usage and utility functions
def main():
    """Example usage of the DataCollector"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bug Bounty Data Collector")
    parser.add_argument("--target", default="example-target", help="Target name")
    parser.add_argument("--subdomains", default="subdomains.txt", help="Subdomains file")
    parser.add_argument("--resolved", default="resolved.txt", help="Resolved domains file")
    parser.add_argument("--urls", default="urls.txt", help="URLs file")
    parser.add_argument("--output", default="recon_results.json", help="Output file")
    parser.add_argument("--workers", type=int, default=20, help="Max concurrent workers")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout")
    parser.add_argument("--rate-limit", type=float, default=0.1, help="Rate limit (seconds)")
    
    args = parser.parse_args()
    
    # Create and use data collector
    with DataCollector(
        target_name=args.target,
        max_workers=args.workers,
        timeout=args.timeout,
        rate_limit=args.rate_limit
    ) as collector:
        
        # Collect all data
        results = collector.collect_all_data(
            subdomains_file=args.subdomains,
            resolved_file=args.resolved,
            urls_file=args.urls
        )
        
        # Save results
        collector.save_results(results, args.output)
        
        # Print summary
        print(f"\n=== Reconnaissance Summary for {results.target} ===")
        print(f"Subdomains discovered: {len(results.subdomains)}")
        print(f"Resolved IPs: {len(results.resolved_ips)}")
        print(f"Total endpoints tested: {results.total_endpoints}")
        print(f"Live endpoints: {len([ep for ep in results.live_endpoints if ep.is_live])}")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()