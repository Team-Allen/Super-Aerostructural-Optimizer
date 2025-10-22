"""
UIUC Airfoil Database Integration for PhysicsNeMo Training
=========================================================

Download and integrate real airfoil data from UIUC database for training.
This will use REAL airfoil geometries instead of just theoretical NACA profiles.
"""

import requests
import numpy as np
import os
import re
from urllib.parse import urljoin
import time
from bs4 import BeautifulSoup
import pandas as pd

class UIUCAirfoilDatabase:
    """Interface to UIUC Airfoil Coordinate Database"""
    
    def __init__(self, cache_dir="uiuc_airfoils"):
        self.base_url = "https://m-selig.ae.illinois.edu/ads/"
        self.coord_url = urljoin(self.base_url, "coord_database.html")
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        self.airfoil_list = []
        self.downloaded_airfoils = {}
        
        print(f"🛩️  UIUC Airfoil Database Interface")
        print(f"   Cache directory: {cache_dir}")
    
    def scrape_airfoil_list(self, max_airfoils=100):
        """Scrape the list of available airfoils from UIUC database"""
        
        print(f"🔍 Scraping UIUC airfoil database...")
        print(f"   URL: {self.coord_url}")
        
        try:
            # Get the main page
            response = requests.get(self.coord_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find airfoil coordinate links
            airfoil_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.dat') and 'coord' in href:
                    airfoil_name = os.path.basename(href).replace('.dat', '')
                    airfoil_links.append({
                        'name': airfoil_name,
                        'url': urljoin(self.base_url, href),
                        'filename': href
                    })
            
            # Limit number of airfoils for practical training
            self.airfoil_list = airfoil_links[:max_airfoils]
            
            print(f"✅ Found {len(self.airfoil_list)} airfoils")
            print(f"   Examples: {[a['name'] for a in self.airfoil_list[:5]]}")
            
            return self.airfoil_list
            
        except Exception as e:
            print(f"❌ Failed to scrape airfoil list: {e}")
            
            # Fallback: Use known popular airfoils from UIUC
            popular_airfoils = [
                'clarky', 'e63', 'naca0012', 'naca2412', 'naca4412', 'naca6412',
                'ag24', 'ag25', 'ag26', 'ah79100c', 'ah81k144', 'aquilax',
                'b737a', 'b737b', 'be50', 'bacxxx', 'clarkt', 'dae11',
                'dae21', 'dae31', 'du84132', 'du86084', 'du93w210',
                'e193', 'e205', 'e387', 'e420', 'fx60100', 'fx61184',
                'goe07k', 'goe14k', 'goe15k', 'hq17', 'hq34', 'hq35',
                'hq41', 'ht05', 'ht08', 'ht12', 'ht15', 'ht23',
                'ls0413', 'ls0417', 'mh32', 'mh45', 'mh60', 'naca0006',
                'naca0008', 'naca0009', 'naca0010', 'naca0015', 'naca0018',
                'naca1408', 'naca1410', 'naca1412', 'naca2414', 'naca4415',
                'naca23012', 'naca23015', 'naca64a010', 'naca65010',
                'rae2822', 's1210', 's1223', 's3021', 's4061', 's8036',
                'bacnlf', 'bw050209', 'du84132v', 'hor04', 'hor07',
                'hor12', 'hor20', 'hq17', 'hs1404', 'hs1430',
                'n0012sc', 'n60', 'n63', 'n64', 'nasa0012',
                'nasasc2-0714', 'nrel408c', 'r140', 'rg14', 'rg15',
                's1020', 's1223rtl', 'sd7062', 'selig1223', 'selig5010',
                'whitcomb', 'wortmann', 'fx63137', 'eppler387', 'eppler420'
            ]
            
            self.airfoil_list = [
                {
                    'name': name,
                    'url': f"{self.base_url}coord/{name}.dat",
                    'filename': f"coord/{name}.dat"
                }
                for name in popular_airfoils[:max_airfoils]
            ]
            
            print(f"📋 Using fallback list: {len(self.airfoil_list)} popular airfoils")
            return self.airfoil_list
    
    def download_airfoil(self, airfoil_info, retry_count=3):
        """Download a single airfoil coordinate file"""
        
        name = airfoil_info['name']
        url = airfoil_info['url']
        cache_file = os.path.join(self.cache_dir, f"{name}.dat")
        
        # Check if already cached
        if os.path.exists(cache_file):
            return self.parse_airfoil_file(cache_file)
        
        print(f"📥 Downloading {name}...")
        
        for attempt in range(retry_count):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                
                # Save to cache
                with open(cache_file, 'w') as f:
                    f.write(response.text)
                
                # Parse and return coordinates
                return self.parse_airfoil_file(cache_file)
                
            except Exception as e:
                print(f"   ⚠️ Attempt {attempt+1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(1)
                else:
                    print(f"   ❌ Failed to download {name}")
                    return None
    
    def parse_airfoil_file(self, filename):
        """Parse UIUC airfoil coordinate file format (handles multiple formats)"""
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines and find coordinates
            coord_lines = []
            skip_header = True
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip text header lines
                if skip_header and not line[0].replace('-', '').replace('.', '').replace(' ', '').isdigit():
                    continue
                
                skip_header = False
                
                # Try to parse as x y coordinates (take only first 2 values)
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        # UIUC format validation: x should be [0,1], y should be reasonable
                        if -0.5 <= x <= 1.5 and -1.0 <= y <= 1.0:
                            coord_lines.append([x, y])
                    except ValueError:
                        continue
            
            if len(coord_lines) < 10:
                return None, None
            
            coords = np.array(coord_lines)
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            
            # Ensure x is in [0, 1] range
            x_min, x_max = x_coords.min(), x_coords.max()
            if x_max > x_min:
                x_coords = (x_coords - x_min) / (x_max - x_min)
            
            # Normalize y coordinates to reasonable airfoil scale
            y_range = y_coords.max() - y_coords.min()
            if y_range > 1.0:  # Likely in percentage or wrong scale
                y_coords = y_coords / 100.0  # Convert from percentage
            
            # Ensure y coordinates are reasonable for airfoil (typically -0.2 to +0.2)
            if abs(y_coords.max()) > 1.0 or abs(y_coords.min()) > 1.0:
                y_coords = y_coords / max(abs(y_coords.max()), abs(y_coords.min()))
                y_coords *= 0.15  # Scale to typical airfoil thickness
            
            # Sort by x coordinate for proper ordering
            sort_idx = np.argsort(x_coords)
            x_coords = x_coords[sort_idx]
            y_coords = y_coords[sort_idx]
            
            return {
                'name': os.path.basename(filename).replace('.dat', ''),
                'x_coords': x_coords,
                'y_coords': y_coords,
                'n_points': len(x_coords)
            }
            
        except Exception as e:
            print(f"   ❌ Failed to parse {filename}: {e}")
            return None
    
    def download_airfoil_database(self, max_airfoils=50, batch_size=10):
        """Download multiple airfoils from UIUC database"""
        
        print(f"\n🚀 DOWNLOADING UIUC AIRFOIL DATABASE")
        print(f"   Target: {max_airfoils} airfoils")
        print(f"   Batch size: {batch_size}")
        
        # Get airfoil list
        if not self.airfoil_list:
            self.scrape_airfoil_list(max_airfoils)
        
        successful_downloads = 0
        failed_downloads = 0
        
        for i, airfoil_info in enumerate(self.airfoil_list[:max_airfoils]):
            print(f"\n📦 Batch {i//batch_size + 1}, Airfoil {i+1}/{max_airfoils}")
            
            airfoil_data = self.download_airfoil(airfoil_info)
            
            if airfoil_data:
                self.downloaded_airfoils[airfoil_data['name']] = airfoil_data
                successful_downloads += 1
                print(f"   ✅ {airfoil_data['name']}: {airfoil_data['n_points']} points")
            else:
                failed_downloads += 1
            
            # Small delay to be respectful to server
            if (i + 1) % batch_size == 0:
                print(f"   ⏸️  Batch complete, waiting 2s...")
                time.sleep(2)
        
        print(f"\n🎉 DOWNLOAD COMPLETE!")
        print(f"   ✅ Successful: {successful_downloads}")
        print(f"   ❌ Failed: {failed_downloads}")
        print(f"   📊 Success rate: {successful_downloads/(successful_downloads+failed_downloads)*100:.1f}%")
        
        return self.downloaded_airfoils
    
    def get_random_airfoil(self):
        """Get a random airfoil from downloaded database"""
        
        if not self.downloaded_airfoils:
            return None
        
        name = np.random.choice(list(self.downloaded_airfoils.keys()))
        return self.downloaded_airfoils[name]
    
    def save_database(self, filename="uiuc_airfoil_database.json"):
        """Save downloaded airfoils to JSON file"""
        
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        save_data = {}
        for name, data in self.downloaded_airfoils.items():
            save_data[name] = {
                'name': data['name'],
                'x_coords': data['x_coords'].tolist(),
                'y_coords': data['y_coords'].tolist(),
                'n_points': data['n_points']
            }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"💾 Database saved: {filename}")
        print(f"   {len(save_data)} airfoils saved")
    
    def load_database(self, filename="uiuc_airfoil_database.json"):
        """Load airfoils from JSON file"""
        
        import json
        
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            save_data = json.load(f)
        
        # Convert lists back to numpy arrays
        for name, data in save_data.items():
            self.downloaded_airfoils[name] = {
                'name': data['name'],
                'x_coords': np.array(data['x_coords']),
                'y_coords': np.array(data['y_coords']),
                'n_points': data['n_points']
            }
        
        print(f"📂 Database loaded: {filename}")
        print(f"   {len(self.downloaded_airfoils)} airfoils available")
        return True

def integrate_uiuc_with_training():
    """Integrate UIUC database with PhysicsNeMo training"""
    
    print(f"\n🎯 INTEGRATING UIUC DATABASE WITH PHYSICSNEMO TRAINING")
    
    # Initialize UIUC database
    uiuc_db = UIUCAirfoilDatabase()
    
    # Try to load existing database first
    if uiuc_db.load_database():
        print("✅ Using cached UIUC database")
    else:
        print("📥 Downloading fresh UIUC database...")
        uiuc_db.download_airfoil_database(max_airfoils=30, batch_size=5)
        uiuc_db.save_database()
    
    # Show statistics
    if uiuc_db.downloaded_airfoils:
        print(f"\n📊 UIUC DATABASE STATISTICS:")
        print(f"   Total airfoils: {len(uiuc_db.downloaded_airfoils)}")
        
        # Point count statistics
        point_counts = [data['n_points'] for data in uiuc_db.downloaded_airfoils.values()]
        print(f"   Point counts: {np.min(point_counts)}-{np.max(point_counts)} (avg: {np.mean(point_counts):.0f})")
        
        # Sample airfoil names
        sample_names = list(uiuc_db.downloaded_airfoils.keys())[:10]
        print(f"   Sample airfoils: {sample_names}")
        
        # Test random sampling
        print(f"\n🎲 TESTING RANDOM AIRFOIL SAMPLING:")
        for i in range(3):
            airfoil = uiuc_db.get_random_airfoil()
            if airfoil:
                print(f"   {i+1}. {airfoil['name']}: {airfoil['n_points']} points")
        
        return uiuc_db
    else:
        print("❌ No airfoils available in database")
        return None

def main():
    """Test UIUC database integration"""
    
    print("🛩️  UIUC AIRFOIL DATABASE INTEGRATION TEST")
    
    # Test the integration
    uiuc_db = integrate_uiuc_with_training()
    
    if uiuc_db and uiuc_db.downloaded_airfoils:
        print(f"\n🎉 SUCCESS! UIUC database integrated")
        print(f"   Ready for PhysicsNeMo training with REAL airfoil geometries")
        print(f"   This will make the neural network much more robust!")
        
        # Show how to use in training
        print(f"\n💡 USAGE IN TRAINING:")
        print(f"   - Replace NACA generation with: airfoil = uiuc_db.get_random_airfoil()")
        print(f"   - Use real coordinates: x_coords = airfoil['x_coords']")
        print(f"   - Train on diverse geometries instead of just NACA profiles")
        
        return True
    else:
        print(f"\n❌ UIUC integration failed")
        print(f"   Falling back to NACA profiles for training")
        return False

if __name__ == "__main__":
    main()