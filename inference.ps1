# Define the JSON data payload for the request body
$jsonPayload = @"
{
  "img1": "img/pc.jpg",
  "img2": "img/test-1.png",
  "img": "img/pc.jpg",
  "tgt_width": 640,
  "tgt_height": 640
}
"@

# Define the target URL (assuming your Docker container is running on localhost)
$url = "http://localhost:8080/sam"

# Send the POST request with the JSON body
Invoke-RestMethod -Uri $url -Method Post -Body $jsonPayload -ContentType "application/json"