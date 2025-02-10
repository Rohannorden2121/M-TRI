#!/bin/bash
# M-TRI GitHub Deployment Script
# This script will create the GitHub repository and push everything automatically

set -e  # Exit on any error

echo "🚀 M-TRI GitHub Deployment Starting..."
echo "=================================="

# Check if we're in the right directory
if [[ ! -f "README.md" ]] || [[ ! -d "src" ]]; then
    echo "❌ Error: Please run this script from the M-TRI project root directory"
    exit 1
fi

# Check if git is configured
if ! git config --get user.name > /dev/null; then
    echo "⚠️  Git user not configured. Please run:"
    echo "   git config --global user.name 'Your Name'"
    echo "   git config --global user.email 'your.email@example.com'"
    exit 1
fi

echo "📋 Step 1: Authenticating with GitHub..."
if ! gh auth status > /dev/null 2>&1; then
    echo "🔐 Please authenticate with GitHub:"
    gh auth login --web
else
    echo "✅ Already authenticated with GitHub"
fi

echo ""
echo "📋 Step 2: Creating GitHub repository..."
REPO_NAME="m-tri"
DESCRIPTION="M-TRI: Machine learning system for predicting harmful algal blooms in NJ waterbodies using satellite imagery and environmental data"

# Check if repository already exists
if gh repo view "$REPO_NAME" > /dev/null 2>&1; then
    echo "⚠️  Repository '$REPO_NAME' already exists. Using existing repository..."
else
    echo "🆕 Creating new repository '$REPO_NAME'..."
    gh repo create "$REPO_NAME" \
        --description "$DESCRIPTION" \
        --public \
        --source=. \
        --remote=origin \
        --push
fi

echo ""
echo "📋 Step 3: Setting up repository topics and settings..."
gh repo edit --add-topic machine-learning \
             --add-topic environmental-science \
             --add-topic water-quality \
             --add-topic harmful-algal-blooms \
             --add-topic satellite-imagery \
             --add-topic fastapi \
             --add-topic streamlit \
             --add-topic docker \
             --add-topic new-jersey \
             --add-topic public-health \
             --add-topic geospatial \
             --add-topic python

echo ""
echo "📋 Step 4: Verifying repository setup..."
REPO_URL=$(gh repo view --json url -q .url)
echo "✅ Repository created successfully!"
echo "🔗 Repository URL: $REPO_URL"

echo ""
echo "📋 Step 5: Checking GitHub Actions..."
echo "🔄 GitHub Actions will automatically run when you push code"
echo "   - Tests will run on every push"
echo "   - Docker images will be built"
echo "   - Code quality checks will be performed"

echo ""
echo "🎉 SUCCESS! Your M-TRI project is now on GitHub!"
echo "=================================="
echo ""
echo "📊 Repository Stats:"
echo "   📁 Files: $(find . -type f | wc -l | tr -d ' ')"
echo "   💾 Size: $(du -sh . | cut -f1)"
echo "   🐍 Python files: $(find . -name '*.py' | wc -l | tr -d ' ')"
echo "   📝 Documentation files: $(find . -name '*.md' | wc -l | tr -d ' ')"
echo ""
echo "🔗 Quick Links:"
echo "   🌐 Repository: $REPO_URL"
echo "   🚀 Actions: $REPO_URL/actions"
echo "   📋 Issues: $REPO_URL/issues"
echo "   📈 Insights: $REPO_URL/pulse"
echo ""
echo "💡 Next Steps:"
echo "   1. ⭐ Star your repository to showcase it"
echo "   2. 📌 Pin it to your GitHub profile"
echo "   3. 📱 Add a social preview image in Settings"
echo "   4. 🔗 Share the link in your applications/resume"
echo ""
echo "🏆 Your M-TRI project is now ready to impress admissions officers!"