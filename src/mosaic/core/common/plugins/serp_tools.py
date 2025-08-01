import os
import json
import logging
from typing import Dict, Optional, List
from serpapi import GoogleSearch
import requests


class BasePerplexityClient:
    def __init__(self):
        self.api_key: Optional[str] = None
        self.url = "https://api.perplexity.ai/chat/completions"
        
    @classmethod
    def init(cls, api_key: Optional[str] = None) -> 'BasePerplexityClient':
        instance = cls()
        instance.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not instance.api_key:
            raise ValueError("API key required via init() or PERPLEXITY_API_KEY env var")
        return instance
        
    def get_base_params(self, prompt: str, **kwargs) -> Dict:
        return {
            "model": kwargs.get("model", "sonar"),
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.9),
            "search_domain_filter": kwargs.get("domains", []),
            "return_images": kwargs.get("images", False),
            "search_recency_filter": kwargs.get("recency", "month"),
            "top_k": kwargs.get("top_k", 0),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "frequency_penalty": kwargs.get("frequency_penalty", 1)
        }
        
    def process_results(self, results: Dict) -> Dict:
        return {
            "text": results["choices"][0]["message"]["content"],
            "citations": results.get("citations", []),
            "images": results["choices"][0]["message"].get("images", [])
        }
        
    def ask(self, prompt: str, **kwargs) -> Dict:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.url, 
                json=self.get_base_params(prompt, **kwargs),
                headers=headers
            )
            response.raise_for_status()
            return self.process_results(response.json())
            
        except Exception as e:
            print(f"Error in ask(): {str(e)}")
            return {}




class BaseSerpSearch:
    """Base class for SERP API searches"""
    
    def __init__(self, query: str, limit: int = 10, **kwargs):
        self.query = query
        self.limit = limit
        self.kwargs = kwargs
        self.output: List[Dict] = []
        
    def get_base_params(self) -> Dict:
        """Get common parameters for all SERP searches"""
        return {
            "api_key": os.environ["SERPAPI_API_KEY"],
            "engine": self.engine,
            "q": self.query,
            "h1": "en",
            "gl": "us", 
            "google_domain": "google.com",
            "num": str(self.limit)
        }
    
    def process_results(self, results: Dict) -> List[Dict]:
        """Process raw results into standardized format"""
        raise NotImplementedError("Subclasses must implement process_results")
        
    def run(self) -> 'BaseSerpSearch':
        """Execute the search and process results"""
        try:
            params = {**self.get_base_params(), **self.get_search_params()}
            print (params)
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if self.results_key not in results:
                print(f"No {self.results_key} found in results")
                return self
                
            self.output = self.process_results(results)
            
        except Exception as e:
            print(f"Error in {self.__class__.__name__}: {str(e)}")
            self.output = []
            
        return self

class NewsSearch(BaseSerpSearch):
    """Class for Google News searches"""
    
    def __init__(self, query: str, limit: int = 10, num_weeks: int = 1):
        super().__init__(query, limit)
        self.num_weeks = num_weeks
        self.engine = "google"
        self.results_key = "news_results"
        
    def get_search_params(self) -> Dict:
        """Get news-specific search parameters"""
        return {
            "tbm": "nws",
            "tbs": f"qdr:w{self.num_weeks}"
        }
        
    def process_results(self, results: Dict) -> List[Dict]:
        self.titles_and_links = [{'title': x['title'], 'link': x['link']} for x in results[self.results_key]]
        return [{
            'title': result['title'],
            'link': result['link'],
            'snippet': result.get('snippet', '')
        } for result in results[self.results_key]]

class OrganicSearch(BaseSerpSearch):
    """Class for Google organic searches"""
    
    def __init__(self, query: str, limit: int = 10):
        super().__init__(query, limit)
        self.engine = "google"
        self.results_key = "organic_results"
        
    def get_search_params(self) -> Dict:
        """Get organic search-specific parameters"""
        return {}
        
    def process_results(self, results: Dict) -> List[Dict]:
        self.titles_and_links = [{'title': x['title'], 'link': x['link']} for x in results[self.results_key]]
        return [{
            'title': result['title'],
            'link': result['link'],
            'snippet': result.get('snippet', '')
        } for result in results[self.results_key]]

class JobsSearch(BaseSerpSearch):
    """Class for Google Jobs searches"""
    
    def __init__(self, query: str, limit: int = 10, location: Optional[str] = None):
        super().__init__(query, limit, location=location)
        self.engine = "google_jobs"
        self.results_key = "jobs_results"
        
    def get_search_params(self) -> Dict:
        """Get jobs-specific search parameters"""
        params = {}
        if self.kwargs.get('location'):
            params['location'] = self.kwargs['location']
        return params
        
    def process_results(self, results: Dict) -> List[Dict]:
        print (results[self.results_key][0])
        self.titles_and_links = [{'title': x['title'], 'link': x['share_link']} for x in results[self.results_key]]
        
        return [{
            'title': result.get('title', ''),
            'company_name': result.get('company_name', ''),
            'location': result.get('location', ''),
            'link': result.get('share_link', ''),
            'description': result.get('description', ''),
            'posted_at': result.get('detected_extensions', {}).get('posted_at', ''),
            'salary': result.get('detected_extensions', {}).get('salary', ''),
            'job_type': result.get('detected_extensions', {}).get('schedule_type', ''),
            'via': result.get('via', ''),
            'qualifications': result.get('detected_extensions', {}).get('qualifications', ''),
            'job_highlights': result.get('job_highlights', []),
            'apply_options': result.get('apply_options', [])
        } for result in results[self.results_key]]



def search_news(query: str, limit: int = 5, num_weeks: int = 2) -> dict:
    """
    this tool takes a query, limit, and num_weeks and returns a list of news article links
    that will have news articles from the last num_weeks weeks related to the query
    Args:
        query: Search query string
        limit: Maximum number of results (default 5)
        num_weeks: Number of weeks to look back (default 2)
    Returns:
        list: A list of news article links
    """
    try :
        import sys, os
        sys.path.append("/tmp/")
    except Exception as e:
        print(f"Error importing sys: {e}")
        return {"error": str(e)}
    
    try :
        from securitygpt.agents.experts.search.utils.serp_utils import NewsSearch
        news_search = NewsSearch(query=query, limit=limit, num_weeks=num_weeks)
        results = news_search.run().output
    except Exception as e:
        print(f"Error importing NewsSearch: {e}")
        return {"error": str(e)}
    
    return {"results": results}


def search_organic(query: str, limit: int = 5) -> dict:
    """
    Search organic web results using SERP API
    Args:
        query: Search query string
        limit: Maximum number of results (default 5)
    Returns:
        Dictionary containing organic search results
    """
    import sys
    sys.path.append("/tmp/")
    
    
    organic_search = OrganicSearch(query=query, limit=limit)
    results = organic_search.run().output
    return {"results": results}