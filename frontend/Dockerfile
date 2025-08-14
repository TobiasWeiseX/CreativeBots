# Node runtime image to use
FROM node:20-alpine AS builder

# Install npm gloabally
RUN npm install -g npm

# Set production
ENV NODE_ENV production

# Add a work directory
WORKDIR /app

# Cache and Install dependencies
# Copy package.json and package-lock.json to the working directory so the can be found to start the app
COPY package*.json ./

# Install the application dependencies
RUN npm i

# Copy app files
COPY . .

# Build the app
# RUN npm run build:mercedes
RUN npm run build

# Bundle static assets with nginx
FROM nginx:1.25-alpine as production

RUN adduser --uid 5000 --shell /bin/sh --system nextsearch \
    && chown -R nextsearch /var/log/nginx \
    && chown -R nextsearch /var/cache/nginx \
    && chown -R nextsearch /etc/nginx

# Add your nginx.conf
COPY --chown=nextsearch nginx.conf /etc/nginx/nginx.conf

# Copy built assets from builder
COPY --from=builder --chown=nextsearch ../build /usr/share/nginx/html

# Expose port
EXPOSE 3000

USER nextsearch

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
